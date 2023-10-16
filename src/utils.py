import os
import pickle
import pandas as pd
import numpy as np
import torch
import sklearn.model_selection
import wandb
import cmat
import src.resamplers
import torchmetrics


def grid_search(args):
    """
    Wrapper around sklearn's parameter grid. Mends
    dict values that are not wrapped in list
    """
    args = args if isinstance(args, (list, tuple)) else [args]
    return sklearn.model_selection.ParameterGrid([
        {k: v if isinstance(v, (list, tuple)) else [v] for k, v in a.items()}
        for a in args
    ])

def get_filenames_of_dataset(ds_path, common_pattern=None, filetype='.csv'):
    '''Returns training files/folders

    Parameters
    ----------
    ds_path: str
    common_pattern: str, optional
       Some part of the filenames that all have in common
    filtype: str, optional
        E.g., file ending or "dir" for directories

    Returns
    -------
    : list

    '''
    res = os.listdir(ds_path)
    if common_pattern:
        res = [x for x in res if common_pattern in x]
    _res = [x for x in res if filetype in x]
    if len(_res)==0:
        _res = [x for x in res if os.path.isdir(os.path.join(ds_path,x))]
    res = _res
    return res


def cv_split(data, folds, randomize=0, split_p=None):
    """
    Do a cross validation split on subjects
    """
    assert folds!=1 or split_p, 'Provide test split percentage "split_p" if folds==1'
    if folds > len(data):
        raise ValueError(f'More folds than subjects provided {folds} > {len(data)}')
    # Do leave-one-out if fold is zero or a negative number
    if folds <= 0:
        folds = len(data)
    # Make a list of subjects and do a seeded shuffle if configured
    subjects = list(data)
    if randomize > 0:
        np.random.seed(randomize)
        np.random.shuffle(subjects)
    if folds > 1:
        # Get step size and loop over folds
        step = int(np.ceil(len(data) / folds))
    else:
        # In case only 1 fold required it is splitted according to split_p
        step = int(np.ceil(len(data) * split_p))
    for fold in range(folds):
        valid = subjects[fold * step:(fold + 1) * step]
        train = [s for s in subjects if not s in valid]
        yield fold, train, valid


def store_tensor(x, path):
    '''Stores given tensor as csv on disk'''
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    np.savetxt(path, x.numpy())


def load_tensor(path):
    '''Loads given file to torch.Tensor'''
    return torch.tensor(np.loadtxt(path), dtype=torch.float32)


def argmax(t, axis=-1):
    '''Computes argmax of tensor or dict of tensors'''
    if type(t) == dict:
        return {k: np.argmax(v, axis=axis) for k, v in t.items()}
    else:
        return np.argmax(t, axis=axis)


def compute_cmat(y_true, y_pred, labels, names,
                 y_true_probs=None, y_pred_probs=None,
                 additional_metrics=[]):
    '''Creates cmat(s) for tensors or dict of tensors

    Parameters
    ----------
    y_true (tensor or dict of tensors)
    y_pred (tensor or dict of tensors)
    labels (list): Labels that can occur in y_true/y_pred
    names (dict of str): For each class in y_true/y_pred, a name.
    y_true_probs (tensor or dict of tensors): y_true but in probabilities
        instead of absolute values
    y_pred_probs (tensor or dict of tensors): y_pred but in probabilities
        instead of absolute values
    additional_metrics (list of str):
        In case additional metrics are needed that are not already
        in the cmat package

    Returns
    -------
    cmat.ConfusionMatrix or dict of cmat.ConfusionMatrix

    '''
    if type(y_true)==dict:
        cmats = {}
        for yt_key, yt_val in y_true.items():
            yp_val = y_pred[yt_key]
            yt_probs_val = y_true_probs[yt_key] if y_true_probs else None
            yp_probs_val = y_pred_probs[yt_key] if y_pred_probs else None
            ccm = create_extended_cmat(
                y_true = yt_val,
                y_pred = yp_val,
                labels = labels,
                names = names,
                y_true_probs = yt_probs_val,
                y_pred_probs = yp_probs_val,
                additional_metrics = additional_metrics
            )
            cmats[yt_key] = ccm
        return cmats
    else:
        ccm = create_extended_cmat(
            y_true = y_true,
            y_pred = y_pred,
            labels = labels,
            names = names,
            y_true_probs = y_true_probs,
            y_pred_probs = y_pred_probs,
            additional_metrics = additional_metrics
        )
        return ccm


def create_extended_cmat(y_true, y_pred, labels=None, names=None,
                         y_true_probs=None, y_pred_probs=None,
                         additional_metrics=[]):
    '''Keeps y_true,y_pred as instance vars of ConfusionMatrix object

    Also stores additional metrics into ConfusionMatrix object in case
    they do not exist already in the cmat package

    '''
    o = cmat.ConfusionMatrix.create(y_true, y_pred, labels=labels, names=names)
    o.y_true = y_true
    o.y_pred = y_pred
    allowed_additional_metrics = ['KLD', 'MAE', 'average_MAE']
    for am in additional_metrics:
        if am not in allowed_additional_metrics:
            raise ValueError(f'Metric {am} unknown. Allowed additional metrics: {allowed_additional_metrics}')
        if am == 'KLD':
            assert y_true_probs is not None and y_pred_probs is not None
            o.KLD = float(torchmetrics.functional.kl_divergence(
                p=torch.tensor(y_true_probs),
                q=torch.tensor(y_pred_probs),
                log_prob=True,
                reduction='mean'
            ).numpy())
        if am == 'MAE':
            assert y_true_probs is not None and y_pred_probs is not None
            o.MAE = abs(y_true_probs-y_pred_probs).mean(axis=0)
            o.MAE = pd.Series(o.MAE, names)
        if am == 'average_MAE':
            assert y_true_probs is not None and y_pred_probs is not None
            o.average_MAE = abs(y_true_probs-y_pred_probs).mean()
    return o


def get_score(cmat, metric):
    '''Metric of given cmat(s)

    Parameters
    ----------
    cmat (ConfusionMatrix or dict of ConfusionMatrix)
    metric (str)

    Returns
    -------
    float
        If cmat is dict, average metric is computed

    '''
    if type(cmat)==dict:
        scores = []
        for cm in cmat.values():
            scores.append(getattr(cm, metric))
        return np.mean(scores)
    else:
        return getattr(cmat, metric)



def save_intermediate_cmat(path, filename, args, cmats,
                           valid_subjects=None):
    '''Save cmat object and args in pickle file'''
    if type(cmats)==dict:
        for _, v in cmats.items():
            # Delete y_true/y_pred created in extended_cmat to save disk space
            if hasattr(v, 'y_true'): delattr(v, 'y_true')
            if hasattr(v, 'y_pred'): delattr(v, 'y_pred')
    else:
        if hasattr(cmats, 'y_pred'): delattr(cmats, 'y_pred')
        if hasattr(cmats, 'y_true'): delattr(cmats, 'y_true')
    if valid_subjects is None:
        args_cmats = [args, cmats]
    else:
        # In case subject filenames are provided:
        args_cmats = [args, cmats, valid_subjects]
    if not os.path.exists(path):
        os.makedirs(path)
    filehandler = open(path+filename, 'wb')
    pickle.dump(args_cmats, filehandler)
    filehandler.close()


def get_existing_args(cmat_path):
    '''Existing arguments of a grid search'''
    existing_arguments = []
    if os.path.exists(cmat_path):
        for cmat_file in os.listdir(cmat_path):
            if cmat_file.endswith('.pkl'):
                with open(os.path.join(cmat_path, cmat_file), 'rb') as f:
                    existing_arguments.append(pickle.load(f)[0])
    return existing_arguments


def args_exist(args, ds_args, cmat_path):
    existing_arguments = get_existing_args(cmat_path)
    all_args = {}
    all_args.update(args)
    all_args.update(ds_args)
    ee = []
    for ii in range(len(existing_arguments)):
        e = all(existing_arguments[ii].get(key, None) == val for key, val in all_args.items())
        ee.append(e)
    return any(ee)


def wandb_init(run_name, wandb_config, entity, proj_name, key='<PutWANDBKeyHere>'):
    wandb.login(key=key)
    rr = wandb.init(
        project=proj_name,
        entity=entity,
        config=wandb_config
    )
    rr.name = run_name + '(' + rr.name + ')'


def log_wandb_cmat(
    y_true,
    y_pred,
    class_names,
    log_name
):
    '''Logs confusion matrix in wandb

    Parameters
    ----------
    y_true: array like
    y_pred: array like
    class_names: list of str
        for each class index a name
    log_name: str

    '''
    cmat_name = 'cmat_' + log_name
    wandb.log({cmat_name: wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true, preds=y_pred,
                    class_names=class_names,
                    title=cmat_name)})
    print(f'Logged {cmat_name}')


def log_cmat_metrics_to_wandb(
    log_cmat,
    log_name,
    class_names,
    metrics=['average_f1score', 'cmat']
):
    '''Log ConfusionMatrix metrics to wandb

    Parameters
    ----------
    log_cmat (cmat.ConfusionMatrix):
        Important: has to be created with src.utils.create_extended_cmat
        such that it includes y_true and y_pred instances
    log_name (str)
    class_names (list of str): For each class index a name
    metrics (list of str): Which metrics to log

    '''
    allowed_metrics = ['f1score',
                       'average_f1score',
                       'recall',
                       'average_recall',
                       'precision',
                       'average_precision',
                       'accuracy',
                       'cmat',
                       'KLD',
                       'MAE',
                       'average_MAE']
    assert set(metrics)<=set(allowed_metrics), print(f'{allowed_metrics}')
    assert ('cmat' not in metrics) or \
           ('cmat' in metrics and \
             hasattr(log_cmat, 'y_true') and \
             hasattr(log_cmat, 'y_pred')
           ), print('Create cmat with src.utils.create_extended_cmat')
    for metric in metrics:
        if metric == 'cmat':
            log_wandb_cmat(
                y_true=log_cmat.y_true,
                y_pred=log_cmat.y_pred,
                class_names=class_names,
                log_name=log_name
            )
        elif metric in ['average_f1score',
                        'average_precision',
                        'average_recall',
                        'accuracy',
                        'KLD',
                        'average_MAE']:
            try:
                to_log = getattr(log_cmat, metric)
                wandb.log({f'{metric}_{log_name}': to_log})
            except AttributeError as e:
                print(e)
        else:
            try:
                data = [(l,m) for l,m in getattr(log_cmat, metric).items()]
                table = wandb.Table(data=data, columns=['label', metric])
                wandb.log({f'{metric}_{log_name}': wandb.plot.bar(
                    table, 'label', metric, title=f'bar_{metric}_{log_name}')}
                )
            except AttributeError as e:
                print(e)


def log_history_metrics_to_wandb(metrics_dict, log_name, metrics=None):
    '''Logs dict of metrics to wandb as lineplots

    Parameters
    ----------
    metrics_dict (dict of list): key=metric name, value=list of metric vals
    log_name (str)
    metrics (list of str): Which of the keys to log. None=all

    '''
    metrics = metrics if metrics else metrics_dict.keys()
    for m in metrics:
        if m not in metrics_dict.keys(): continue
        data = [(i, e) for i,e in enumerate(metrics_dict[m])]
        table = wandb.Table(data=data, columns=['iteration', m])
        wandb.log({f'{m}_{log_name}_history': wandb.plot.line(table,
                   'iteration', m, title=f'{m}_{log_name}_history')})

class MomentEstimator:
    """
    Running moment estimator

    TODO: is this numerically stable for very large n?
    """

    def __init__(self, shape, moments):
        self.shape = shape
        self.moments = moments
        self.exp = torch.Tensor(
            np.array(moments).reshape(-1, *(1,)*len(shape))
        )
        self.data = torch.zeros((len(moments), *shape),dtype=torch.float64)
        self.n = 0

    def update(self, data):
        assert len(data.shape) >= 1
        n = data.shape[0]
        assert data.shape == (n, *self.data.shape[1:])
        a = self.n / (self.n + n)
        self.data = a * self.data + (1 - a) * data.mean(axis=0) ** self.exp
        self.n += n

    def get(self, moment):
        return self.data[self.moments.index(moment)]

    def __repr__(self):
        return f'MomentEstimator(shape={self.shape}, moments={self.moments} n={self.n})'

    @classmethod
    def merge(cls, ests):
        new_est = cls(ests[0].shape, ests[0].moments)
        ns = [est.n for est in ests]
        new_est.n = sum(ns)
        new_est.data = sum(est.data * (est.n / new_est.n) for est in ests)
        return new_est

    def save(self, path):
        np.savez(path, shape=self.shape, moments=self.moments, data=self.data, n=self.n)

    @classmethod
    def load(cls, path):
        archive = np.load(path)
        est = cls(tuple(archive['shape']), list(archive['moments']))
        est.n = int(archive['n'])
        est.data = archive['data']
        return est


def complex_to_cartesian(x):
    """Converts a complex valued tensor to 2D cartesian coordinates

    Parameters
    ----------
    x: torch.tensor<...>
        Complex valued tensor (e.g. raw stfts)

    Returns
    -------
    torch.tensor<..., 2>
        The same tensor with one higher rank with the real component
        of the input in [..., 0] and the imaginary component in [..., 1]
    """
    return torch.stack([torch.real(x), torch.imag(x)], -1)

def complex_to_magnitude(x, expand=False):
    """Takes the magnitude (abs) of a complex valued tensor

    Parameters
    ----------
    x: torch.tensor<...>
        Complex valued tensor (e.g. raw stfts)
    expand: bool
        Whether to expand the rank of the tensor to one rank higher
        (in order to stay consistent with `complex_to_cartesian`)

    Returns
    -------
    torch.tensor<..., 1> or torch.tensor<...>
        The same tensor, except real-valued and with the absolute value of
        the complex values. If expand true, then a new axis will be created.
        If not, then the shape will be the same as the input
    """
    magnitude = torch.abs(x)
    return torch.unsqueeze(magnitude, -1) if expand else magnitude


def resample(
    signal,
    source_rate,
    target_rate,
    discrete_columns=[],
    timestamp_columns=[],
    resampler='fourier',
    padder=None,
    pad_size=0,
):
    '''Start resampling

    Parameters
    ----------
    signal (pandas.DataFrame): input signal
    source_rate (int): Sample rate of the input signal
    target_rate (int): Target sample rate after resampling
    discrete_columns (list of str), optional: Columns that will just be
        resampled by getting the closest value, default is []
    timestamp_columns (list of str), optional: Timestamp columns that will
        be resampled by interpolation, default is []
    resampler (str), optional: Which resampling method to use
        Choices (decimate, fourier), default is 'fourier'
    padder (padder), optional: If set, pad windows before resampling.
        Does nothing unless the --pad-size argument is also set.
        Choices (inv_reflect, reflect, repeat, wrap, zero, None),
        default is None
    pad_size (int), optional: Number of datapoints to pad with,
        default is 0

    Returns
    -------
    (pandas.DataFrame): resampled signal

    '''
    resampled_signal = src.resamplers.resample(
        signal,
        source_rate=source_rate,
        target_rate=target_rate,
        resampler=resampler,
        padder=padder,
        pad_size=pad_size,
        discrete_columns=discrete_columns,
        timestamp_columns=timestamp_columns
    )
    return resampled_signal
