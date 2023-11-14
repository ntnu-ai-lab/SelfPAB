import sys
import math
import einops
import collections
from collections import OrderedDict

import torch
import torchmetrics
import pytorch_lightning as pl


def get_model(algorithm_name, algorithm_args):
    allowed_models = ['TransformerEncoderNetwork',
                      'DownstreamMLP', 'MLP', 'DeepConvLSTM',
                      'LSTMNetwork', 'CNN']
    if algorithm_name in allowed_models:
        cls = getattr(sys.modules[__name__], algorithm_name)
        return cls(algorithm_args)
    else:
        raise ValueError((f'No algorithm with name "{algorithm_name}".\n'
                          f'Allowed algorithms: {allowed_models}'))

def get_model_class(algorithm_name):
    '''Returns the class without creating an object'''
    allowed_models = ['TransformerEncoderNetwork',
                      'DownstreamMLP', 'MLP', 'DeepConvLSTM',
                      'LSTMNetwork', 'CNN']
    if algorithm_name in allowed_models:
        cls = getattr(sys.modules[__name__], algorithm_name)
        return cls
    else:
        raise ValueError((f'No algorithm with name "{algorithm_name}".\n'
                          f'Allowed algorithms: {allowed_models}'))


###########################################################################
#                 Def DL models for the framework                         #
###########################################################################


class TransformerEncoderNetwork(pl.LightningModule):
    def __init__(self, args):
        '''TransformerEncoder network

        Consisting of input embedding (linear projection),
        positional encoding, transformer encoder layers,
        and final prediction head (MLP)

        '''
        super().__init__()
        # Architecture
        self.emb = InputEmbeddingPosEncoding(args)
        encoder_layer = TransformerEncoderLayer(
            d_model=args['d_model'],
            nhead=args['nhead'],
            dim_feedforward=args['dim_feedforward'],
            dropout=args['dropout'],
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=args['n_encoder_layers']
        )
        self.seq_operation = args['seq_operation'] \
                if 'seq_operation' in args else None
        self.prediction_head = get_pred_head(
            num_layers=args['n_prediction_head_layers'],
            input_dim=args['d_model'],
            output_dim=args['output_dim'],
            hidden_dim=args['dim_prediction_head']
        )
        self.automatic_optimization = False
        self.algorithm_args = args
        self.criterion = get_loss(args['loss'])
        self.output_activation = get_activation(args['output_activation'])
        self.metrics = init_metrics(args['metrics'], args)
        self.val_metrics = init_metrics(args['metrics'], args)
        self.test_metrics = init_metrics(args['metrics'], args)
        self.save_hyperparameters()

    def forward(self, x, output_attention=False):
        x = self.emb(x)
        x = self.transformer_encoder(
            x,
            output_attention=output_attention
        )
        if output_attention:
            x, attention_weights = x
        x = apply_seq_operation(x, self.seq_operation)
        x = self.prediction_head(x)
        y_hat = self.output_activation(x, dim=-1)
        if output_attention:
            return y_hat, attention_weights
        else:
            return y_hat

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        x, y = train_batch
        x = self.emb(x)
        x = self.transformer_encoder(x)
        x = apply_seq_operation(x, self.seq_operation)
        y_hat = self.prediction_head(x)
        loss = self.criterion(y_hat, y)
        ##### Optimization #####
        opt.zero_grad()
        self.manual_backward(loss)
        sch = self.lr_schedulers()
        opt.step()
        if sch is not None: sch.step()
        ##### Logging #####
        self.log('train_loss', loss, prog_bar=True)
        self.log_dict({'lr': opt.param_groups[0]['lr']}, prog_bar=False)
        self._log_metrics(y, y_hat, self.metrics, step_call='train')

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = self.emb(x)
        x = self.transformer_encoder(x)
        x = apply_seq_operation(x, self.seq_operation)
        y_hat = self.prediction_head(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self._log_metrics(y, y_hat, self.val_metrics, step_call='val')

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = self.emb(x)
        x = self.transformer_encoder(x)
        x = apply_seq_operation(x, self.seq_operation)
        y_hat = self.prediction_head(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        self._log_metrics(y, y_hat, self.test_metrics, step_call='test')

    def _log_metrics(self, y, y_hat, metrics_list, step_call):
        for metric in metrics_list:
            if type(metric) == torchmetrics.KLDivergence:
                # Compute softmax for pred to get probs
                y_hat_prob = self.output_activation(y_hat, dim=1)
                # Stack batches and sequences for KLD
                metric(einops.rearrange(y, 'N S C -> (N S) C'),
                       einops.rearrange(y_hat_prob, 'N S C -> (N S) C'))
            else:
                metric(einops.rearrange(y_hat, 'N S C -> N C S'), y)
            self.log(
                step_call+'_'+str(metric),
                metric,
                prog_bar=False,
                on_step=False,
                on_epoch=True
            )

    def configure_optimizers(self):
        return config_optimizers(self.algorithm_args, self.parameters)


class DownstreamMLP(pl.LightningModule):
    def __init__(self, args):
        '''MLP with given upstream model'''
        super().__init__()
        # Architecture
        rw = args['random_weight_init'] if 'random_weight_init' in args \
                                        else False
        # Multibranching if needed
        self.upstream_models = torch.nn.ModuleList()
        self.upstream_branches = args['upstream_branches'] \
                if 'upstream_branches' in args else 1
        self.weight_sharing = args['weight_sharing'] \
                if 'weight_sharing' in args else False
        self.branch_merge_operation = args['branch_merge_operation'] \
                if 'branch_merge_operation' in args else 'max'
        # For weight_sharing only 1 upstream model is used but repetitively
        _actual_upstream_branches = 1 \
                if self.weight_sharing else self.upstream_branches
        for ub in range(_actual_upstream_branches):
            self.upstream_models.extend([UpstreamModel(
                algorithm_name=args['upstream_algorithm'],
                model_path=args['upstream_model_path'],
                cut_layers=args['rmv_upstream_layers'],
                weighted_sum_layer=args['weighted_sum_layer'],
                random_weight_init=rw
            )])
        for um in self.upstream_models: um.freeze()
        # When to unfreeze in percent of total step count
        self.fine_tune_step = args['fine_tune_step']
        self.total_step_count = args['total_step_count']
        assert 0 <= self.fine_tune_step <= 1
        ph_dim = self._get_upstream_output_dim(input_dim=args['input_dim'])
        self.prediction_head = get_pred_head(
            num_layers=args['n_prediction_head_layers'],
            input_dim=ph_dim,
            hidden_dim=args['dim_prediction_head'],
            output_dim=args['output_dim']
        )
        self.automatic_optimization = False
        self.algorithm_args = args
        self.criterion = get_loss(args['loss'])
        self.output_activation = get_activation(args['output_activation'])
        self.metrics = init_metrics(args['metrics'], args)
        self.val_metrics = init_metrics(args['metrics'], args)
        self.save_hyperparameters()
        self.n_steps = 0

    def forward(self, x):
        x = self.upstream_feature_extraction(x)
        x = self.prediction_head(x)
        y_hat = self.output_activation(x, dim=-1)
        return y_hat

    def upstream_feature_extraction(self, x):
        splitted_x = []
        if type(x)!=list:
            x_sub_shapes = x.shape[-1]//self.upstream_branches
            # Split input channels evenly across all upstream models
            for idx in range(self.upstream_branches):
                # In case of weight_sharing only 1 upstream model exists
                # which is used repetitively
                upstream_model = self.upstream_models[0] \
                        if self.weight_sharing else self.upstream_models[idx]
                splitted_x.extend(
                    [upstream_model(x[:,:,idx*x_sub_shapes:(idx+1)*x_sub_shapes])]
                )
        else:  # x with timestamps
            x, x_ts = x
            x_sub_shapes = x.shape[-1]//self.upstream_branches
            for idx in range(self.upstream_branches):
                upstream_model = self.upstream_models[0] \
                        if self.weight_sharing else self.upstream_models[idx]
                splitted_x.extend(
                    [upstream_model([x[:,:,idx*x_sub_shapes:(idx+1)*x_sub_shapes], x_ts])]
                )
        if self.branch_merge_operation == 'max':
            # Max pool for the multi-branches
            x = torch.max(torch.stack(splitted_x,dim=-1), dim=-1)[0]
        elif self.branch_merge_operation == 'mean':
            # Average pool for multi-branches
            x = torch.mean(torch.stack(splitted_x,dim=-1), dim=-1)
        elif self.branch_merge_operation == 'concat':
            # Concat for multi-branches
            x = torch.cat(splitted_x, dim=-1)
        else:
            raise ValueError(f'Unknown branch_merge_operation: {self.branch_merge_operation}')
        return x

    def training_step(self, train_batch, batch_idx):
        self.n_steps += 1
        progress = self.n_steps/self.total_step_count
        if self.upstream_models[0].frozen and progress > self.fine_tune_step:
            print('Unfreeze upstream_model')
            for um in self.upstream_models: um.unfreeze()
        opt = self.optimizers()
        if len(train_batch) == 2:
            x, y = train_batch
            mask = torch.ones(y.shape[:2]).to(y.device)
        else:
            x, y, mask = train_batch
        x = self.upstream_feature_extraction(x)
        y_hat = self.prediction_head(x)
        # Ignore padded labels during loss computation:
        loss = self.criterion(y_hat, y, reduction='none')*mask
        # Compute own mean with mask
        loss = loss.sum()/mask.sum()
        ##### Optimization #####
        opt.zero_grad()
        self.manual_backward(loss)
        sch = self.lr_schedulers()
        opt.step()
        if sch is not None: sch.step()
        ##### Logging #####
        self.log('train_loss', loss, prog_bar=True)
        self.log_dict({'lr': opt.param_groups[0]['lr']}, prog_bar=False)
        self._log_metrics(y, y_hat, self.metrics, step_call='train')

    def validation_step(self, val_batch, batch_idx):
        if len(val_batch) == 2:
            x, y = val_batch
            mask = torch.ones(y.shape[:2]).to(y.device)
        else:
            x, y, mask = val_batch
        x = self.upstream_feature_extraction(x)
        y_hat = self.prediction_head(x)
        # Ignore padded labels during loss computation:
        loss = self.criterion(y_hat, y, reduction='none')*mask
        # Compute own mean with mask
        loss = loss.sum()/mask.sum()
        self.log('val_loss', loss, prog_bar=True)
        self._log_metrics(y, y_hat, self.val_metrics, step_call='val')

    def _log_metrics(self, y, y_hat, metrics_list, step_call):
        for metric in metrics_list:
            if type(metric) == torchmetrics.KLDivergence:
                # Compute softmax for pred to get probs
                y_hat_prob = self.output_activation(y_hat, dim=1)
                # Stack batches and sequences for KLD
                metric(einops.rearrange(y, 'N S C -> (N S) C'),
                       einops.rearrange(y_hat_prob, 'N S C -> (N S) C'))
            else:
                metric(einops.rearrange(y_hat, 'N S C -> N C S'), y)
            self.log(
                step_call+'_'+str(metric),
                metric,
                prog_bar=False,
                on_step=False,
                on_epoch=True
            )

    def _get_upstream_output_dim(self, input_dim):
        try:
            ph_dim = self.upstream_models[0](
                torch.rand([1, 1, input_dim//self.upstream_branches])
            ).shape[-1]
        except ValueError:  # In case upstream model assumes timestamps
            ph_dim = self.upstream_models[0]([
                torch.rand([1, 1, input_dim//self.upstream_branches]),
                torch.rand([1, 1, 5])
            ]).shape[-1]
        if self.branch_merge_operation == 'concat':
            ph_dim = ph_dim * self.upstream_branches
        return ph_dim

    def configure_optimizers(self):
        return config_optimizers(self.algorithm_args, self.parameters)


class MLP(pl.LightningModule):
    def __init__(self, args):
        '''Simple MLP'''
        super().__init__()
        self.mlp = get_pred_head(
            num_layers=args['n_layers'],
            input_dim=args['input_dim'],
            hidden_dim=args['dim_hidden'],
            output_dim=args['output_dim']
        )
        self.automatic_optimization = False
        self.algorithm_args = args
        self.criterion = get_loss(args['loss'])
        self.output_activation = get_activation(args['output_activation'])
        self.metrics = init_metrics(args['metrics'], args)
        self.val_metrics = init_metrics(args['metrics'], args)
        self.save_hyperparameters()

    def forward(self, x):
        x = self.mlp(x)
        y_hat = self.output_activation(x, dim=-1)
        return y_hat

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        if len(train_batch) == 2:
            x, y = train_batch
            mask = torch.ones(y.shape[:2]).to(y.device)
        else:
            x, y, mask = train_batch
        y_hat = self.mlp(x)
        # Ignore padded labels during loss computation:
        loss = self.criterion(y_hat, y, reduction='none')*mask
        # Compute own mean with mask
        loss = loss.sum()/mask.sum()
        ##### Optimization #####
        opt.zero_grad()
        self.manual_backward(loss)
        sch = self.lr_schedulers()
        opt.step()
        if sch is not None: sch.step()
        ##### Logging #####
        self.log('train_loss', loss, prog_bar=True)
        self.log_dict({'lr': opt.param_groups[0]['lr']}, prog_bar=False)
        self._log_metrics(y, y_hat, self.metrics, step_call='train')

    def validation_step(self, val_batch, batch_idx):
        if len(val_batch) == 2:
            x, y = val_batch
            mask = torch.ones(y.shape[:2]).to(y.device)
        else:
            x, y, mask = val_batch
        y_hat = self.mlp(x)
        # Ignore padded labels during loss computation:
        loss = self.criterion(y_hat, y, reduction='none')*mask
        # Compute own mean with mask
        loss = loss.sum()/mask.sum()
        self.log('val_loss', loss, prog_bar=True)
        self._log_metrics(y, y_hat, self.val_metrics, step_call='val')

    def _log_metrics(self, y, y_hat, metrics_list, step_call):
        for metric in metrics_list:
            if type(metric) == torchmetrics.KLDivergence:
                # Compute softmax for pred to get probs
                y_hat_prob = self.output_activation(y_hat, dim=1)
                # Stack batches and sequences for KLD
                metric(einops.rearrange(y, 'N S C -> (N S) C'),
                       einops.rearrange(y_hat_prob, 'N S C -> (N S) C'))
            else:
                metric(einops.rearrange(y_hat, 'N S C -> N C S'), y)
            self.log(
                step_call+'_'+str(metric),
                metric,
                prog_bar=False,
                on_step=False,
                on_epoch=True
            )

    def configure_optimizers(self):
        return config_optimizers(self.algorithm_args, self.parameters)


class DLBaseline(pl.LightningModule):
    def __init__(self, args):
        '''Superclass for different deep learning baseline classifier'''
        super().__init__()
        self.seq_operation = args['seq_operation'] \
                if 'seq_operation' in args else None
        self.automatic_optimization = False
        self.algorithm_args = args
        self.criterion = get_loss(args['loss'])
        self.output_activation = get_activation(args['output_activation'])
        self.metrics = init_metrics(args['metrics'], args)
        self.val_metrics = init_metrics(args['metrics'], args)
        self.test_metrics = init_metrics(args['metrics'], args)

    def forward(self, x, apply_out_activation=True):
        msg = ('Implement individual forward()')
        raise NotImplementedError(msg)

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        if len(train_batch) == 2:
            x, y = train_batch
            mask = torch.ones(y.shape[:2]).to(y.device)
        else:
            x, y, mask = train_batch
        y_hat = self.forward(x, apply_out_activation=False)
        y = self._post_proc_y(y)
        if self.seq_operation is not None:
            loss = self.criterion(y_hat, y)
        else:
            # Ignore padded labels during loss computation:
            loss = self.criterion(y_hat, y, reduction='none')*mask
            # Compute own mean with mask
            loss = loss.sum()/mask.sum()
        ##### Optimization #####
        opt.zero_grad()
        self.manual_backward(loss)
        sch = self.lr_schedulers()
        opt.step()
        if sch is not None: sch.step()
        ##### Logging #####
        self.log('train_loss', loss, prog_bar=True)
        self.log_dict({'lr': opt.param_groups[0]['lr']}, prog_bar=True)
        self._log_metrics(y, y_hat, self.metrics, step_call='train')

    def validation_step(self, val_batch, batch_idx):
        if len(val_batch) == 2:
            x, y = val_batch
            mask = torch.ones(y.shape[:2]).to(y.device)
        else:
            x, y, mask = val_batch
        y_hat = self.forward(x, apply_out_activation=False)
        y = self._post_proc_y(y)
        if self.seq_operation is not None:
            loss = self.criterion(y_hat, y)
        else:
            # Ignore padded labels during loss computation:
            loss = self.criterion(y_hat, y, reduction='none')*mask
            # Compute own mean with mask
            loss = loss.sum()/mask.sum()
        self.log('val_loss', loss, prog_bar=True)
        self._log_metrics(y, y_hat, self.val_metrics, step_call='val')

    def _log_metrics(self, y, y_hat, metrics_list, step_call):
        for metric in metrics_list:
            if type(metric) == torchmetrics.KLDivergence:
                # Compute softmax for pred to get probs
                y_hat_prob = self.output_activation(y_hat, dim=1)
                # Stack batches and sequences for KLD
                metric(einops.rearrange(y, 'N S C -> (N S) C'),
                       einops.rearrange(y_hat_prob, 'N S C -> (N S) C'))
            else:
                if len(y_hat.shape)==3:
                    metric(einops.rearrange(y_hat, 'N S C -> N C S'), y)
                else:
                    metric(y_hat, y)
            self.log(
                step_call+'_'+str(metric),
                metric,
                prog_bar=False,
                on_step=False,
                on_epoch=True
            )

    def _post_proc_y(self, y):
        '''Depending on seq_operation, different ground truths

        This is only applied during train or valid, not during test

        If None, nothing is done
        If last, last y in sequence is ground truth
        Else, majority voting

        '''
        if self.seq_operation is None:
            return y
        elif self.seq_operation == 'last':
            # Last as ground truth
            return apply_seq_operation(y, operation_name=self.seq_operation)
        else:
            # Majority voting for window
            return torch.mode(y)[0]

    def configure_optimizers(self):
        return config_optimizers(self.algorithm_args, self.parameters)


class DeepConvLSTM(DLBaseline):
    def __init__(self, args):
        '''Reimplementation of the DeepConvLSTM

        Model first proposed in:
        [F. J. Ordóñez and D. Roggen, “Deep Convolutional and LSTM
         Recurrent Neural Networks for Multimodal Wearable Activity
         Recognition,” Sensors, vol. 16, no. 1, Art. no. 1, Jan. 2016,
         doi: 10.3390/s16010115.]

        '''
        super().__init__(args)
        self.cnn = _CNN(
            num_layers=args['n_cnn_layers'],
            input_dim=args['input_dim'],
            num_filters=args['num_filters'],
            kernel_size=args['kernel_size'],
            padding=args['cnn_pad'] if 'cnn_pad' in args else 0
        )
        self.lstm = torch.nn.LSTM(
            input_size=args['num_filters'],
            hidden_size=args['hidden_lstm_dim'],
            num_layers=args['n_lstm_layers'],
            batch_first=True,
            dropout=args['dropout']
        )
        # Single Dense layer for classes
        self.prediction_head = get_pred_head(
            num_layers=args['n_prediction_head_layers'],
            input_dim=args['hidden_lstm_dim'],
            output_dim=args['output_dim'],
            hidden_dim=args['dim_prediction_head']
        )
        self.save_hyperparameters()

    def forward(self, x, apply_out_activation=True):
        # For CNN, sequence length S has to be last dimension
        x = einops.rearrange(x, 'N S C -> N C S')
        x = self.cnn(x)
        # For LSTM undo prev. rearrange
        x = einops.rearrange(x, 'N C S -> N S C')
        x, _ = self.lstm(x)
        # last seq element of LSTM is used for prediction
        x = apply_seq_operation(x, operation_name=self.seq_operation)
        x = self.prediction_head(x)
        if apply_out_activation:
            x = self.output_activation(x, dim=-1)
        return x


class CNN(DLBaseline):
    def __init__(self, args):
        '''Simple Convolutional Neural Network'''
        super().__init__(args)
        self.cnn = _CNN(
            num_layers=args['n_cnn_layers'],
            input_dim=args['input_dim'],
            num_filters=args['num_filters'],
            kernel_size=args['kernel_size'],
            padding=args['cnn_pad'] if 'cnn_pad' in args else 0
        )
        ph_dim = self._get_cnn_output_dim(args['input_dim'],
                                          args['sequence_length'])
        # Single Dense layer for classes
        self.prediction_head = get_pred_head(
            num_layers=args['n_prediction_head_layers'],
            input_dim=ph_dim,
            output_dim=args['output_dim'],
            hidden_dim=args['dim_prediction_head']
        )
        self.save_hyperparameters()

    def forward(self, x, apply_out_activation=True):
        # For CNN, sequence length S has to be last dimension
        x = einops.rearrange(x, 'N S C -> N C S')
        x = self.cnn(x)
        # Undo prev. rearrange
        x = einops.rearrange(x, 'N C S -> N S C')
        x = apply_seq_operation(x, operation_name=self.seq_operation)
        x = self.prediction_head(x)
        if apply_out_activation:
            x = self.output_activation(x, dim=-1)
        return x

    def _get_cnn_output_dim(self, input_dim, seq_len):
        cnn_out = self.cnn(torch.rand([1, input_dim, seq_len])).shape
        if self.seq_operation=='flatten':
            return cnn_out[-2]*cnn_out[-1]
        else:
            return cnn_out[-2]


class LSTMNetwork(DLBaseline):
    def __init__(self, args):
        '''Simple LSTM Network'''
        super().__init__(args)
        self.lstm = torch.nn.LSTM(
            input_size=args['input_dim'],
            hidden_size=args['hidden_lstm_dim'],
            num_layers=args['n_lstm_layers'],
            batch_first=True,
            dropout=args['dropout']
        )
        # Single Dense layer for classes
        self.prediction_head = get_pred_head(
            num_layers=args['n_prediction_head_layers'],
            input_dim=args['hidden_lstm_dim'],
            output_dim=args['output_dim'],
            hidden_dim=args['dim_prediction_head']
        )
        self.save_hyperparameters()

    def forward(self, x, apply_out_activation=True):
        x, _ = self.lstm(x)
        x = apply_seq_operation(x, operation_name=self.seq_operation)
        x = self.prediction_head(x)
        if apply_out_activation:
            x = self.output_activation(x, dim=-1)
        return x


###########################################################################
#            Helpful globally accessible functions/classes                #
###########################################################################
class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class LinearRelu(pl.LightningModule):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        '''Simple linear layer followed by a relu activation and dropout'''
        super().__init__()
        linear = torch.nn.Linear(
            in_features=input_dim,
            out_features=output_dim
        )
        relu = torch.nn.ReLU()
        if dropout>0.0:
            dropout_layer = torch.nn.Dropout(dropout)
            self.linear_relu = torch.nn.Sequential(linear, relu, dropout_layer)
        else:
            self.linear_relu = torch.nn.Sequential(linear, relu)

    def forward(self, x):
        return self.linear_relu(x)


class Conv1DRelu(pl.LightningModule):
    def __init__(self, input_dim, num_filters, kernel_size, stride=1, padding=0, dropout=0.0):
        '''Simple Conv1D layer followed by a relu activation'''
        super().__init__()
        conv1d = torch.nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        relu = torch.nn.ReLU()
        if dropout>0.0:
            dropout_layer = torch.nn.Dropout(dropout)
            self.conv1d_relu = torch.nn.Sequential(conv1d, relu, dropout_layer)
        else:
            self.conv1d_relu = torch.nn.Sequential(conv1d, relu)

    def forward(self, x):
        return self.conv1d_relu(x)


class _MLP(pl.LightningModule):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''MLP implementation

        and output layer with linear activation

        '''
        super().__init__()
        layers = OrderedDict()
        _lastdim = input_dim
        for i in range(num_layers-1):
            layers[f'layer{i}'] = LinearRelu(input_dim=_lastdim,
                                             output_dim=hidden_dim)
            _lastdim = hidden_dim
        layers['out_layer'] = torch.nn.Linear(
            in_features=_lastdim,
            out_features=output_dim
        )
        self.mlp = torch.nn.Sequential(layers)

    def forward(self, x):
        return self.mlp(x)


class _CNN(pl.LightningModule):
    def __init__(self, num_layers, input_dim, num_filters, kernel_size, padding=0, dropout=0.0):
        '''Convolutional Neural Network implementation

        Sequence of Conv1D layers
        Each Conv1D layer is followed by a ReLU activation
        and dropout if dropout prob > 0

        '''
        super().__init__()
        layers = OrderedDict()
        _lastdim = input_dim
        for i in range(num_layers):
            layers[f'Conv1D_{i}'] = Conv1DRelu(input_dim=_lastdim,
                                               num_filters=num_filters,
                                               kernel_size=kernel_size,
                                               padding=padding,
                                               dropout=dropout)
            _lastdim = num_filters
        self.cnn = torch.nn.Sequential(layers)

    def forward(self, x):
        return self.cnn(x)


class InputEmbeddingPosEncoding(torch.nn.Module):
    def __init__(self, args):
        '''Combined InputEmbedding and PositionalEncoding in one module'''
        super().__init__()
        self.lin_proj_layer = torch.nn.Linear(
            in_features=args['input_dim'],
            out_features=args['d_model']
        )
        self.pos_encoder = get_pos_encoder(
            args['positional_encoding'],
            d_model=args['d_model'],
            dropout=0.0
        )

    def forward(self, x):
        pe_x = self.pos_encoder(x)
        x = self.lin_proj_layer(x[0] if type(x)==list else x)
        return x + pe_x  # Input embedding + positional encoding



class PositionalEncodings(torch.nn.Module):
    def __init__(self, encoder_names, **kwargs):
        '''Sequential of  positionl encodings

        Parameters
        ----------
        encoder_names (list of str): Name of encoders to use
        kwargs (dict): additional encoder specific parameters

        '''
        super().__init__()
        self.pos_encoders = torch.nn.ModuleList()
        for _pe in encoder_names:
            try:
                pe_class = getattr(sys.modules[__name__], _pe)
            except AttributeError as e:
                raise ValueError(f'Unknown possitional encoding: {_pe}')
            self.pos_encoders.extend([
                pe_class(d_model=kwargs['d_model'], dropout=kwargs['dropout'])
            ])

    def forward(self, x):
        '''Sum of all positional encodings'''
        return sum([_pe(x) for _pe in self.pos_encoders])


class FixedEmbedding(torch.nn.Module):
    def __init__(self, c_in, d_model):
        '''General Embedding class for TemporalPositionalEncoding

        Source: https://github.com/zhouhaoyi/Informer2020/blob/main/models/embed.py
        See also: https://doi.org/10.48550/arXiv.2012.07436

        '''
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = torch.nn.Embedding(c_in, d_model)
        self.emb.weight = torch.nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.0):
        '''Positional encoding based on 5-dimensional timestamps

        Timestamps need to be of the form [month, day, weekday, hour, minute]
        Inspired by:
        Source: https://github.com/zhouhaoyi/Informer2020/blob/main/models/embed.py
        See also: https://doi.org/10.48550/arXiv.2012.07436

        '''
        super(TemporalPositionalEncoding, self).__init__()
        minute_size = 60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        self.minute_embed = FixedEmbedding(minute_size, d_model)
        self.hour_embed = FixedEmbedding(hour_size, d_model)
        self.weekday_embed = FixedEmbedding(weekday_size, d_model)
        self.day_embed = FixedEmbedding(day_size, d_model)
        self.month_embed = FixedEmbedding(month_size, d_model)

    def forward(self, x):
        '''x needs to be tuple containing as second entry the timestamps'''
        _, x_ts = x
        x_ts = x_ts.long()
        minute_x = self.minute_embed(x_ts[:,:,4])
        hour_x = self.hour_embed(x_ts[:,:,3])
        weekday_x = self.weekday_embed(x_ts[:,:,2])
        day_x = self.day_embed(x_ts[:,:,1])
        month_x = self.month_embed(x_ts[:,:,0])
        return minute_x + hour_x + weekday_x + day_x + month_x


class AbsolutePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=10000, batch_first=True):
        '''Sinusoidal positional encoding

        Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        Parameters
        ----------
        d_model (int): input dimension
        dropout (float):  random zeroes input elements with probab. dropout
        max_len (int): max sequence length
        batch_first (bool): If True, then the input and output tensors are
            provided as (batch, seq, feature). Default: True (batch, seq, feature).

        '''
        super().__init__()
        self.dropout_prob = dropout
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = einops.rearrange(pe,'S B D -> B S D')
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
                if batch_first else [batch_size, seq_len, embedding_dim]
        """
        if type(x) == list:
            x, _ = x  # Drop timestamp here
        if not self.batch_first:
            pe = self.pe[:x.size(0)]
        else:
            pe = self.pe[:,:x.size(1)]
        if self.dropout_prob!=0.0:
            return self.dropout(pe)
        else:
            return pe


def get_pos_encoder(encoder_names, **kwargs):
    '''Returns the required positional encoder(s)'''
    if type(encoder_names) != list:
        try:
            pe_class = getattr(sys.modules[__name__], encoder_names)
        except AttributeError as e:
            raise ValueError(f'Unknown possitional encoding: {encoder_names}')
        return pe_class(d_model=kwargs['d_model'], dropout=kwargs['dropout'])
    else:
        return PositionalEncodings(encoder_names, **kwargs)


class UpstreamModel(pl.LightningModule):
    def __init__(self, algorithm_name, model_path,
                 cut_layers, weighted_sum_layer=None,
                 random_weight_init=False):
        '''General purpose upstream model

        Parameters
        ----------
        algorithm_name (str): Used algorithm
        model_path (str): Path to ckpt file
        cut_layers (int): How many layers to cut at end
        weighted_sum_layer (int): Which part of upstream_model for WS comp.
            None=no weighted sum
        random_weight_init (bool): Whether to use the upstream model's
            architecture but init the weights randomly

        '''
        super().__init__()
        self.activation = {}
        self.do_weighted_sum = weighted_sum_layer is not None
        model_cls = get_model_class(algorithm_name)
        cp_model = model_cls.load_from_checkpoint(
            model_path
        )
        if random_weight_init:
            print('Use random weights for upstream model')
            cp_model = model_cls(cp_model.algorithm_args)
        self.upstream_model = []
        for i, l in enumerate(list(cp_model.children())[:-cut_layers]):
            if self.do_weighted_sum and i == weighted_sum_layer:
                _l = [p for p in l.children()][0] \
                     if type(l)==TransformerEncoder \
                     else l
                num_ws_weights = 0
                for j, subl in enumerate(_l.children()):
                    num_ws_weights += 1
                    subl.register_forward_hook(self.get_layer_output(str(j)))
            self.upstream_model.append(l)
        self.upstream_model = torch.nn.Sequential(*self.upstream_model)
        if self.do_weighted_sum:
            self.weighted_sum = LinearWeightedSum(num_ws_weights)
        self.frozen = False

    def get_layer_output(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def forward(self, x):
        x = self.upstream_model(x)
        if self.do_weighted_sum:
            x = self.weighted_sum(list(self.activation.values()))
        return x

    def predict_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, _ = batch
        else:
            x = batch
        return self.forward(x)

    def freeze(self):
        '''Freezes all upstream model weights'''
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        self.frozen = True

    def unfreeze(self):
        '''Unfreezes all upstream model weights'''
        for param in self.parameters():
            param.requires_grad = True
        self.train()
        self.frozen = False


class LinearWeightedSum(torch.nn.Module):
    def __init__(self, n_inputs):
        '''Weighted sum of given inputs'''
        super(LinearWeightedSum, self).__init__()
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.Tensor([1/n_inputs])) for i in range(n_inputs)]
        )

    def forward(self, input):
        return sum([self.weights[i]*input[i] for i in range(len(input))])


class ModuleListWithForward(torch.nn.ModuleList):
    def __init__(self):
        '''ModuleList with forward function

        It assumes that all modules in the list get exactly the same input

        '''
        super(ModuleListWithForward, self).__init__()

    def forward(self, x):
        '''Applies x to each module'''
        return [m(x) for m in self]


class LossList:
    def __init__(self, fct_name_list):
        '''Callable loss function list

        Each function in fct list is expected to receive target,prediction
        and return a float

        Parameters
        ----------
        fct_name_list (list of str)

        '''
        if type(fct_name_list)!=list:
            fct_name_list = [fct_name_list]
        self.fct_list = []
        for loss_name in fct_name_list:
            if loss_name == 'CrossEntropyLoss':
                self.fct_list.append(torch.nn.functional.cross_entropy)
            elif loss_name == 'BinaryCrossEntropyLoss':
                self.fct_list.append(torch.nn.functional.binary_cross_entropy_with_logits)
            elif loss_name == 'L1':
                self.fct_list.append(torch.nn.functional.l1_loss)
            elif loss_name == 'L2':
                self.fct_list.append(torch.nn.functional.mse_loss)
            elif loss_name == 'BCEWithLogitsLoss':
                self.fct_list.append(torch.nn.functional.binary_cross_entropy_with_logits)
            elif loss_name == 'maskedL1':
                reduction = 'mean'  # Maybe consider more than mean
                def masked_l1_loss(output, target, **kwargs):
                    assert len(target) == 2, \
                            'target has to be tuple containing target and mask'
                    target, mask = target[0], target[1]
                    mask_sum = torch.sum(mask, axis=(1,2))
                    mask_sum = torch.where(
                        mask_sum==0,
                        torch.ones_like(mask_sum),
                        mask_sum
                    )  # Avoid div by zero
                    res = (torch.sum(mask*torch.abs(output-target), axis=(1,2)))
                    res = res / mask_sum
                    if reduction == 'mean':
                        return torch.mean(res)
                    else:
                        return res
                self.fct_list.append(masked_l1_loss)
            elif loss_name == 'maskedL2':
                reduction = 'mean'  # Maybe consider more than mean
                def masked_l2_loss(output, target, **kwargs):
                    assert len(target) == 2, \
                            'target has to be tuple containing target and mask'
                    target, mask = target[0], target[1]
                    mask_sum = torch.sum(mask, axis=(1,2))
                    mask_sum = torch.where(
                        mask_sum==0,
                        torch.ones_like(mask_sum),
                        mask_sum
                    )  # Avoid div by zero
                    res = (torch.sum(mask*torch.square(output-target), axis=(1,2)))
                    res = res / mask_sum
                    if reduction == 'mean':
                        return torch.mean(res)
                    else:
                        return res
                self.fct_list.append(masked_l2_loss)
            else:
                raise ValueError(f'Loss {loss_name} not implemented')

    def __call__(self, pred, target, reduction='mean'):
        '''Sums up all losses in fct_list

        pred and target have to have the same number of entries as number
        of loss functions in fct_list

        '''
        if len(self.fct_list)==1:
            loss_fct = self.fct_list[0]
            y_hat = pred
            y = target
            if loss_fct==torch.nn.functional.cross_entropy and len(y_hat.shape)==3:
                y_hat = einops.rearrange(y_hat, 'N S C -> N C S')
                if len(y.shape)==len(y_hat.shape):
                    # reshape required if target provided in probabilities
                    y = einops.rearrange(y, 'N S C -> N C S')
            return loss_fct(y_hat, y, reduction=reduction)
        losses = []
        for i, loss_fct in enumerate(self.fct_list):
            y_hat = pred[i]
            y = target[i]
            # reshape required for correct cross entropy loss computation:
            # [batch_size, seq_len, num_classes] -> [batch_size, num_classes, seq_len]
            if loss_fct==torch.nn.functional.cross_entropy and len(y_hat.shape)==3:
                y_hat = einops.rearrange(y_hat, 'N S C -> N C S')
                if len(y.shape)==len(y_hat.shape):
                    # reshape required if target provided in probabilities
                    y = einops.rearrange(y, 'N S C -> N C S')
            losses.append(loss_fct(y_hat, y, reduction=reduction))
        return sum(losses)


class ActivationList:
    def __init__(self, fct_name_list):
        '''Callable activation function list

        Parameters
        ----------
        fct_name_list (list of str)

        '''
        if type(fct_name_list)!=list:
            fct_name_list = [fct_name_list]
        self.fct_list = []
        for activation_name in fct_name_list:
            if activation_name == 'softmax':
                print('Softmax output activation')
                self.fct_list.append(torch.nn.functional.softmax)
            elif activation_name is None or activation_name=='identity':
                print('No output activation')
                def identity(x, *args, **kwargs):
                    return x
                self.fct_list.append(identity)
            else:
                raise ValueError(f'Activation {activation_name} not implemented')

    def __call__(self, x, dim):
        '''Applies each activation to each x

        x has to have the same number of elements as fct_list

        '''
        if len(self.fct_list) == 1:
            return self.fct_list[0](x, dim=dim)
        return [act(x[i], dim=dim) for i, act in enumerate(self.fct_list)]


###########################################################################
#                        Custom LR Scheduler                              #
###########################################################################
class WarmupSqrtLRSchedule(torch.optim.lr_scheduler._LRScheduler):
    '''Linear warmup and decrease afterwards using inverse sqrt step num

    Source:
    https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py

    '''
    def __init__(self, optimizer, lr_mul, d_model,
                 n_warmup_steps, total_step_count=None):
        self.optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        if n_warmup_steps < 1:
            # n_warmup_steps seen as percentage
            if total_step_count is None:
                raise ValueError('total_step_count cannot be None when ' \
                                 'n_warmup_steps between 0 and 1')
            self.n_warmup_steps = int(n_warmup_steps * total_step_count)
        else:
            self.n_warmup_steps = n_warmup_steps
        print(f'Num Warmup Steps: {self.n_warmup_steps}')
        self.n_steps = 0

    def step(self):
        ''' Learning rate scheduling per step '''
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


class LinearWUSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, initial_learning_rate,
                 decay_steps, end_learning_rate=0, warmup=0):
        self.optimizer = optimizer
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.warmup = warmup
        self.n_steps = 1
        assert 0 <= warmup < 1

    def step(self):
        self.n_steps += 1
        progress = min(self.n_steps/self.decay_steps, 1)
        progress_wu = progress / (self.warmup + 1e-6)
        lr_wu = (
            (1 - progress_wu) * self.end_learning_rate +
            progress_wu * self.initial_learning_rate
        )
        progress_cd = (progress - self.warmup) / (1 - self.warmup)
        lr_cd = (
            (1 - progress_cd) * self.initial_learning_rate +
            progress_cd * self.end_learning_rate
        )
        lr = lr_wu if progress<self.warmup else lr_cd
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class ExponentialSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, init_lr, decay_steps, decay_rate):
        '''Exponential schedule similar to tensorflow's ExponentialDecay'''
        self.optimizer = optimizer
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.init_lr = init_lr
        self.n_steps = 0

    def step(self):
        self.n_steps += 1
        lr = self.init_lr * self.decay_rate**(self.n_steps/self.decay_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def get_loss(loss_name):
    return LossList(loss_name)


def get_activation(activation_name):
    return ActivationList(activation_name)


def apply_seq_operation(x, operation_name, dim=1):
    '''Applies operation on seq dimension

    If None, nothing is done and all seq samples are just forwarded.
    If mean, all seq elements are averaged.
    If max, max pooling on seq dimension
    If last, last element in seq selected
    If flatten, seq is flattened with feature/channel dim

    Parameters
    ----------
    x (tensor): shape [batch_size, seq_len, feature_dim]
    operation_name (str): Name of operation to apply on seq elements.
        None="seq elements just forwarded (no operation applied)"
    dim (int): Which dim is the seq dim (default 1)

    Returns
    -------
    operation(x)

    '''
    if operation_name is None:
        return x
    elif operation_name=='mean':
        return torch.mean(x, dim=dim)
    elif operation_name=='max':
        return torch.max(x, dim=dim)[0]
    elif operation_name=='last':
        return torch.select(x, dim=dim, index=-1)
    elif operation_name=='flatten':
        return torch.flatten(x, start_dim=dim)
    else:
        raise ValueError(f'Seq operation {operation_name} not implemented')


def config_optimizers(args, params):
    ##### Optimizer #####
    opt_name = args['optimizer']
    if opt_name == 'Adam':
        optimizer = torch.optim.Adam(
            params(),
            lr=float(args['lr']),
            betas=(0.9, 0.98),  # As in: Attention Is All You Need
            eps=1e-08,
            weight_decay=float(args['weight_decay'])
        )
    elif opt_name == 'AdamW':
        optimizer = torch.optim.AdamW(
            params(),
            lr=float(args['lr']),
            betas=(0.9, 0.98),  # As in: Attention Is All You Need
            eps=1e-08,
            weight_decay=float(args['weight_decay'])
        )
    elif opt_name == 'SGD':
        optimizer = torch.optim.SGD(
            params(),
            lr=float(args['lr'])
        )
    elif opt_name == 'RMSProp':
        optimizer = torch.optim.RMSprop(
            params(),
            lr=float(args['lr']),
            weight_decay=float(args['weight_decay'])
        )
    elif opt_name == 'Adadelta':
        optimizer = torch.optim.Adadelta(
            params(),
            lr=float(args['lr']),
            weight_decay=float(args['weight_decay'])
        )
    else:
        raise ValueError(f'Optimizer {opt_name} not implemented')
    ##### Lr scheduler #####
    scheduler_name = args['lr_scheduler']
    if scheduler_name is None:
        return optimizer
    elif scheduler_name == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.1,
            verbose=False
        )
    elif scheduler_name == 'ExponentialTFDecay':
        scheduler = ExponentialSchedule(
            optimizer=optimizer,
            init_lr=float(args['lr']),
            decay_steps=args['lr_decay_steps'] if 'lr_decay_steps' in args else 200,
            #decay_steps=args['total_step_count']//args['epochs'],
            #decay_rate=0.9
            decay_rate=args['lr_decay_rate'] if 'lr_decay_rate' in args else 0.8
        )
    elif scheduler_name == 'WarmupSqrtLRSchedule':
        scheduler = WarmupSqrtLRSchedule(
            optimizer=optimizer,
            lr_mul=1.0,
            d_model=args['d_model'],
            n_warmup_steps=0.01,  # First 1% of all steps
            total_step_count=args['total_step_count']
        )
    elif scheduler_name == 'LinearWUSchedule':
        scheduler = LinearWUSchedule(
            optimizer=optimizer,
            initial_learning_rate=float(args['lr']),
            decay_steps=args['total_step_count'],
            end_learning_rate=float(args['lr'])*0.01,
            warmup=0.07
        )
    elif scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args['total_step_count']//args['epochs'],
            gamma=args['lr_decay_rate'] if 'lr_decay_rate' in args else 0.9
        )
    else:
        raise ValueError(f'scheduler {scheduler_name} not implemented')
    return [optimizer], [scheduler]


def init_metrics(metrics, args):
    '''Inits metrics

    Parameters
    ----------
    metrics (list of str)
    **kwargs: additional parameters for metrics

    Returns
    -------
    torch.nn.ModuleList of metrics

    '''
    metrics_modules = torch.nn.ModuleList()
    if metrics is None: return metrics_modules
    if 'Accuracy' in metrics:
        metrics_modules.extend([
            torchmetrics.Accuracy(
                average='micro',
                mdmc_average='global',
                num_classes=args['output_dim']
            )]
        )
    if 'F1Score' in metrics:
        metrics_modules.extend([
            torchmetrics.F1Score(
                average='macro',
                mdmc_average='global',
                num_classes=args['output_dim']
            )]
        )
    if 'Precision' in metrics:
        metrics_modules.extend([
            torchmetrics.Precision(
                average='macro',
                mdmc_average='global',
                num_classes=args['output_dim']
            )]
        )
    if 'Recall' in metrics:
        metrics_modules.extend([
            torchmetrics.Recall(
                average='macro',
                mdmc_average='global',
                num_classes=args['output_dim']
            )]
        )
    if 'KLD' in metrics:
        metrics_modules.extend([
            torchmetrics.KLDivergence(
                log_prob=True,
                reduction='mean'
            )]
        )
    return metrics_modules


def get_pred_head(
    num_layers,
    input_dim=None,
    output_dim=None,
    hidden_dim=None
):
    '''Either an MLP with num_layers layers or Identity if num_layers=0

    If output_dim is a list, a ModuleListWithForward with prediction head
    for each output_dim is returned.

    '''
    if type(output_dim)!=list:
        if num_layers == 0:
            print('No prediction head')
            return torch.nn.Identity()
        else:
            return _MLP(
                num_layers=num_layers,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
    else:
        pred_heads = ModuleListWithForward()
        for out_dim in output_dim:
            if num_layers == 0:
                print('No prediction head')
                pred_heads.extend([torch.nn.Identity()])
            else:
                pred_heads.extend([
                    _MLP(
                        num_layers=num_layers,
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=out_dim
                    )
                ])
        return pred_heads



from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

class MetricsHistoryLogger(LightningLoggerBase):
    def __init__(self):
        super().__init__()
        self.history = collections.defaultdict(list)

    @property
    def name(self):
        return 'MetricsHistoryLogger'

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for metric_name, metric_value in metrics.items():
            if metric_name != 'epoch':
                self.history[metric_name].append(metric_value)
            else:
                if (not len(self.history['epoch']) or
                    not self.history['epoch'][-1] == metric_value):
                    self.history['epoch'].append(metric_value)
                else:
                    pass
        return


class TransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    '''Own TransformerEncoderLayer implementation to return attention weights'''

    def forward(self, src, src_mask=None,
                src_key_padding_mask=None,
                output_attention=False):
        '''Adapted forward to return attention weights if needed'''
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        if (src.dim() == 3 and not self.norm_first and not self.training and
            self.self_attn.batch_first and
            self.self_attn._qkv_same_embed_dim and self.activation_relu_or_gelu and
            self.norm1.eps == self.norm2.eps and
            ((src_mask is None and src_key_padding_mask is None)
             if src.is_nested
             else (src_mask is None or src_key_padding_mask is None))):
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )
            if (not torch.overrides.has_torch_function(tensor_args) and
                    # We have to use a list comprehension here because TorchScript
                    # doesn't support generator expressions.
                    all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]) and
                    (not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]))):
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    False,  # norm_first, currently not supported
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    src_mask if src_mask is not None else src_key_padding_mask,
                )
        x = src
        if self.norm_first:
            sa, aw = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + sa
            x = x + self._ff_block(self.norm2(x))
        else:
            sa, aw = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + sa)
            x = self.norm2(x + self._ff_block(x))

        if output_attention:
            return x, aw
        else:
            return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x, aw = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True,
                           average_attn_weights=False)
        return self.dropout1(x), aw


class TransformerEncoder(torch.nn.TransformerEncoder):
    '''Own TransformerEncoder implementation to return attention weights'''

    def forward(self, src, mask=None, src_key_padding_mask=None, output_attention=False):
        '''Extended forward of nn.TransformerEncoder to return attention_weights if needed'''
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        if isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            if (not first_layer.norm_first and not first_layer.training and
                    first_layer.self_attn.batch_first and
                    first_layer.self_attn._qkv_same_embed_dim and first_layer.activation_relu_or_gelu and
                    first_layer.norm1.eps == first_layer.norm2.eps and
                    src.dim() == 3 and self.enable_nested_tensor) :
                if src_key_padding_mask is not None and not output.is_nested and mask is None:
                    tensor_args = (
                        src,
                        first_layer.self_attn.in_proj_weight,
                        first_layer.self_attn.in_proj_bias,
                        first_layer.self_attn.out_proj.weight,
                        first_layer.self_attn.out_proj.bias,
                        first_layer.norm1.weight,
                        first_layer.norm1.bias,
                        first_layer.norm2.weight,
                        first_layer.norm2.bias,
                        first_layer.linear1.weight,
                        first_layer.linear1.bias,
                        first_layer.linear2.weight,
                        first_layer.linear2.bias,
                    )
                    if not torch.overrides.has_torch_function(tensor_args):
                        if not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]):
                            if output.is_cuda or 'cpu' in str(output.device):
                                convert_to_nested = True
                                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not())
        attention_weights = []
        for mod in self.layers:
            if convert_to_nested:
                output = mod(output, src_mask=mask,
                             output_attention=output_attention)
            else:
                output = mod(output, src_mask=mask,
                             src_key_padding_mask=src_key_padding_mask,
                             output_attention=output_attention)
            if output_attention:
                output, aw = output
                attention_weights.append(aw)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        if output_attention:
            return output, attention_weights
        else:
            return output
