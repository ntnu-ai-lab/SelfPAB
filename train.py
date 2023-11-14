import os
import cmat
import argparse
import random
import numpy as np
import math
import src.config
import src.datasets
import src.utils
import src.models
import src.samplers

import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def train(config, ds_path=None, loso=False):
    """Starts model training with the given config and dataset path

    Parameters
    ----------
    config (src.config.Config)
    ds_path (str)
    loso (bool): Whether a leave-one-out CV is performed

    Returns
    -------
    (pytorch_lightning.LightningModule): (best) trained model
    (cmat.ConfusionMatrix): (best) test results as confusion matrix object
    (dict): (best) all recorded metrics during training in a dict

    """
    # Set all seeds:
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ds_path = config.TRAIN_DATA if ds_path is None else ds_path
    cmat_path = f'{config.CONFIG_PATH}/cmats/'
    if config.VALID_SPLIT=='test':
        valid_subjects = config.TEST_SUBJECTS.copy()
        valid_split = 0.0
    elif type(config.VALID_SPLIT) == list:
        valid_subjects = config.VALID_SPLIT
        valid_split = 0.0
    else:
        valid_subjects = None
        valid_split = config.VALID_SPLIT
    current_iter = 0
    best_model = None
    best_cmat = None
    best_score = None
    best_args = None
    best_logs = None
    # Iterate over all dataset configs if given
    for ds_args in src.utils.grid_search(config.DATASET_ARGS):
        print(f'Dataset arguments: {ds_args}', flush=True)
        # Iterate over all model configs if given
        for args in src.utils.grid_search(config.ALGORITHM_ARGS):
            ######### Train with given args ##########
            print(f'Evaluating arguments: {args}', flush=True)
            if config.SKIP_FINISHED_ARGS and \
               src.utils.args_exist(args, ds_args, cmat_path):
                print(f'Skipping existing args {current_iter}...')
                current_iter += 1
                continue
            # Create the dataset
            skip_files = config.TEST_SUBJECTS.copy()
            if valid_subjects:
                skip_files += valid_subjects
                skip_files = list(set(skip_files))
            dataset = src.datasets.get_dataset(
                dataset_name=config.DATASET,
                dataset_args=ds_args,
                root_dir=ds_path,
                num_classes=config.num_classes,
                label_map=config.label_index,
                replace_classes=config.replace_classes,
                config_path=config.CONFIG_PATH,
                skip_files=skip_files,
                name_label_map=config.class_name_label_map
            )
            # Split into train and validation
            valid_amount = int(np.floor(len(dataset)*valid_split))
            train_amount = len(dataset) - valid_amount
            train_ds, valid_ds = torch.utils.data.random_split(
                dataset,
                lengths=[train_amount, valid_amount],
                generator=torch.Generator().manual_seed(config.SEED)
            )
            if loso or valid_subjects is not None:
                skip_files = [x for x in os.listdir(ds_path) \
                              if x not in valid_subjects]
                valid_ds = src.datasets.get_dataset(
                    dataset_name=config.DATASET,
                    dataset_args=ds_args,
                    root_dir=ds_path,
                    config_path=config.CONFIG_PATH,
                    num_classes=config.num_classes,
                    label_map=config.label_index,
                    replace_classes=config.replace_classes,
                    skip_files=skip_files,
                    valid_mode=True,
                    name_label_map=config.class_name_label_map
                )
                if valid_subjects is not None: print(f'Valid subjects: {valid_subjects}')
            # Create the dataloaders
            collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
            # Subsample train if needed:
            subsample_perc = ds_args['subsample_perc'] \
                    if 'subsample_perc' in ds_args else 1.0
            if subsample_perc == 1.0 and type(subsample_perc)==float:
                sampler = torch.utils.data.RandomSampler(
                    data_source=train_ds,
                    num_samples=len(train_ds)
                )
            else:
                sampler = src.samplers.ActivitySubsetRandomSampler(
                    data_source=train_ds,
                    samples_per_activity=subsample_perc,
                    num_acts=config.num_classes
                )
            train_dl = torch.utils.data.DataLoader(
                dataset=train_ds,
                batch_size=args['batch_size'],
                sampler=sampler,
                num_workers=config.NUM_WORKERS,
                collate_fn=collate_fn
            )
            valid_dl = torch.utils.data.DataLoader(
                dataset=valid_ds,
                batch_size=args['batch_size'],
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                collate_fn=collate_fn
            )
            _epochs = args['epochs']
            total_step_count = len(train_dl)*_epochs
            val_after_nth_step = args['val_after_nth_step'] if 'val_after_nth_step' in args else 100
            val_check_interval = val_after_nth_step/len(train_dl)
            if val_check_interval <= 1:
                check_val_every_n_epoch = 1
            else:
                check_val_every_n_epoch = int(val_check_interval)
                val_check_interval = 1.0
            args.update({'input_dim': dataset.feature_dim,
                         'output_dim': dataset.output_shapes,
                         'total_step_count': total_step_count,
                         'sequence_length': ds_args['sequence_length']})
            print('Create the model')
            model = src.models.get_model(
                algorithm_name=config.ALGORITHM,
                algorithm_args=args
            )
            # Init a trainer and fit
            # Stores all metrics in a dict
            history_logger = src.models.MetricsHistoryLogger()
            loggers = [history_logger]
            if config.WANDB and not loso:
                ds_name = os.path.realpath(ds_path).split('/')[-1]
                proj_name = 'harth_plus_dl_TRAIN_'+config.PROJ_NAME+ds_name
                wandb_logger = WandbLogger(project=proj_name)
                wandb_logger.watch(model, log_graph=False)
                wandb.config.update(ds_args)
                wandb.config.update(args)
                wandb.config.update({'Algorithm': config.ALGORITHM,
                                     'Dataset': config.DATASET,
                                     'Train_DS_size': len(dataset)})
                loggers.append(wandb_logger)
            callbacks = []
            if config.EARLY_STOPPING:
                callbacks.append(EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    min_delta=0.00,
                    patience=10,
                    verbose=True
                    )
                )
            trainer = pl.Trainer(
                gpus=config.NUM_GPUS,
                logger=loggers,
                checkpoint_callback=False,
                max_epochs=args['epochs'],
                log_every_n_steps=5,
                check_val_every_n_epoch=check_val_every_n_epoch,
                val_check_interval=val_check_interval,
                callbacks=callbacks
            )
            trainer.fit(model, train_dl, valid_dl)
            ######### Final Test of given args #########
            if len(config.TEST_SUBJECTS) != 0:
                # Skip the train subjects
                skip_files = [x for x in os.listdir(ds_path) \
                              if x not in config.TEST_SUBJECTS]
                if valid_subjects:
                    for vs in valid_subjects:
                        if vs not in config.TEST_SUBJECTS:
                            skip_files.append(vs)
                test_dataset = src.datasets.get_dataset(
                    dataset_name=config.DATASET,
                    dataset_args=ds_args,
                    root_dir=ds_path,
                    config_path=config.CONFIG_PATH,
                    num_classes=config.num_classes,
                    label_map=config.label_index,
                    replace_classes=config.replace_classes,
                    skip_files=skip_files,
                    test_mode=True, inference_mode=True,
                    name_label_map=config.class_name_label_map
                )
                test_dl = torch.utils.data.DataLoader(
                    dataset=test_dataset,
                    #batch_size=args['batch_size'],
                    batch_size=1,
                    shuffle=False,
                    num_workers=config.NUM_WORKERS
                )
                y_hat = trainer.predict(model, test_dl)
                try:
                    y_hat = torch.cat(y_hat)  # Stack batches
                except RuntimeError as e:
                    print(e)
                # post-process preds
                y_hat_probs, y_true_probs = None, None
                if config.METRIC_AGGR_WINDOW_LEN:
                    y_hat_probs = test_dataset.post_proc_y(
                        t=y_hat,
                        return_probs=True,
                        probs_aggr_window_len=config.METRIC_AGGR_WINDOW_LEN
                    )
                    y_true_probs = test_dataset.y(
                        return_probs=True,
                        probs_aggr_window_len=config.METRIC_AGGR_WINDOW_LEN
                    )
                y_hat = test_dataset.post_proc_y(y_hat)
                y_true = test_dataset.y()  # True label
                # Compute test cmat
                cm = src.utils.compute_cmat(
                    y_true = y_true,
                    y_pred = y_hat,
                    labels = config.possible_indices,
                    names = config.class_names,
                    y_true_probs = y_true_probs,
                    y_pred_probs = y_hat_probs,
                    additional_metrics = config.ADDITIONAL_EVAL_METRICS
                )
                # Save cmat object and args in pickle file:
                cmat_args = args.copy()
                cmat_args.update(ds_args.copy())
                cmat_args.update({'algorithm': config.ALGORITHM,
                                  'dataset': config.DATASET})
                if config.STORE_CMATS:
                    cm_cp = src.utils.compute_cmat(
                        y_true = y_true,
                        y_pred = y_hat,
                        labels = config.possible_indices,
                        names = config.class_names,
                        y_true_probs = y_true_probs,
                        y_pred_probs = y_hat_probs,
                        additional_metrics = config.ADDITIONAL_EVAL_METRICS
                    )
                    src.utils.save_intermediate_cmat(
                        path=cmat_path,
                        filename='args_'+str(current_iter).zfill(6)+'.pkl',
                        args=cmat_args,
                        cmats=cm_cp,
                        valid_subjects=config.TEST_SUBJECTS
                    )
                score = src.utils.get_score(cm, config.EVAL_METRIC)
                if config.WANDB and not loso:
                    wandb.log({f'Test_{config.EVAL_METRIC}': score})
                    wandb.finish()
                if best_score is None or score > best_score:
                    best_model = model
                    best_cmat = cm
                    best_score = score
                    best_logs = history_logger.history
                    best_args = cmat_args
            current_iter += 1
    print(f'Best score: {best_score}, with best args: {best_args}')
    return best_model, best_cmat, best_logs, best_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start ML training.')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='/param/config.yml')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default=None)
    args = parser.parse_args()
    config_path = args.params_path
    # Read config
    config = src.config.Config(config_path)
    ds_path = args.dataset_path
    train(config, ds_path)
