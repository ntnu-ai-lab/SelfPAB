import os
import argparse
import torch
import numpy as np
import random
import src.config
import src.datasets
import src.models
import pytorch_lightning as pl
import wandb
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def upstream(config, ds_path=None):
    """Starts upstream training with the given config and dataset path

    The upstream task is defined in the given config

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
    # Iterate over all model configs if given
    for args in src.utils.grid_search(config.ALGORITHM_ARGS):
        for ds_args in src.utils.grid_search(config.DATASET_ARGS):
            print(f'Dataset arguments: {ds_args}', flush=True)
            ######### Train with given args ##########
            print(f'Evaluating arguments: {args}', flush=True)
            # Create the datasets
            valid_dataset = src.datasets.get_dataset(
                dataset_name=config.DATASET,
                dataset_args=ds_args,
                root_dir=ds_path,
                config_path=config.CONFIG_PATH,
                test_mode=False,
                valid_mode=True,
                skip_files=[]
            )
            test_dataset = src.datasets.get_dataset(
                dataset_name=config.DATASET,
                dataset_args=ds_args,
                root_dir=ds_path,
                config_path=config.CONFIG_PATH,
                test_mode=True, valid_mode=False,
                skip_files=valid_dataset.used_files
            )
            skip_files = valid_dataset.used_files+test_dataset.used_files
            train_dataset = src.datasets.get_dataset(
                dataset_name=config.DATASET,
                dataset_args=ds_args,
                root_dir=ds_path,
                config_path=config.CONFIG_PATH,
                test_mode=False, valid_mode=False,
                skip_files=skip_files
            )
            # Create the dataloaders
            collate_fn = train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None
            valid_dl = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=args['batch_size'],
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                collate_fn=collate_fn
            )
            train_dl = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args['batch_size'],
                shuffle=True,
                num_workers=config.NUM_WORKERS,
                collate_fn=collate_fn
            )
            test_dl = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=args['batch_size'],
                shuffle=False,
                num_workers=1,
                collate_fn=collate_fn
            )
            _epochs = args['epochs']
            total_step_count = len(train_dl)*_epochs
            val_after_nth_step = len(train_dl)/4  # 4 times valid per epoch
            val_check_interval = val_after_nth_step/len(train_dl)
            if val_check_interval <= 1:
                check_val_every_n_epoch = 1
            else:
                check_val_every_n_epoch = int(val_check_interval)
                val_check_interval = 1.0
            print(f'Epochs: {_epochs},   ',
                  f'Steps: {total_step_count},  ',
                  f'val_check_interval: {val_check_interval}',
                  f'check_val_every_n_epoch: {check_val_every_n_epoch}'
                 )
            #######################
            args.update({'input_dim': train_dataset.input_shape,
                         'output_dim': train_dataset.output_shapes,
                         'total_step_count': total_step_count,
                         '_epochs': _epochs})
            print('Create the model')
            model = src.models.get_model(
                algorithm_name=config.ALGORITHM,
                algorithm_args=args
            )
            loggers = []
            if config.WANDB:
                if type(ds_path)==list:
                    ds_name = 'Combined'
                else:
                    ds_name = os.path.realpath(ds_path).split('/')[-1]
                proj_name = 'harth_plus_dl_upstream_'+config.PROJ_NAME+'_'+ds_name
                wandb_logger = WandbLogger(project=proj_name)
                wandb_logger.watch(model, log_graph=False)
                loggers.append(wandb_logger)
            callbacks = []
            cp_path = config.CONFIG_PATH
            cp_name = str(
                datetime.today()
            ).replace(':','_').replace(' ','__').replace('.','_')
            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss' if len(valid_dataset)!=0 else None,
                dirpath=cp_path,
                filename=cp_name,
                verbose=True
            )
            callbacks.append(checkpoint_callback)
            early_stopping = EarlyStopping(
                monitor='val_loss',
                min_delta=0.0,
                patience=total_step_count*0.10,  # 10% of total steps
                verbose=True
            )
            callbacks.append(early_stopping)
            trainer = pl.Trainer(
                gpus=config.NUM_GPUS,
                logger=loggers,
                callbacks=callbacks,
                max_epochs=_epochs,
                num_sanity_val_steps=1,
                log_every_n_steps=1,
                check_val_every_n_epoch=check_val_every_n_epoch,
                val_check_interval=val_check_interval,
                accelerator="gpu",
                strategy='ddp'
            )
            trainer.fit(model, train_dl, valid_dl)
            ##### Final testing #####
            if len(test_dataset) != 0:
                model_cls = src.models.get_model_class(config.ALGORITHM)
                best_model = model_cls.load_from_checkpoint(
                    os.path.join(cp_path,cp_name+'.ckpt')
                )
                best_model.eval()  # eval mode
                trainer.test(best_model, dataloaders=test_dl)
            if config.WANDB: wandb.finish()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start Upstream training.')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='/param/config.yml')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default=None)
    args = parser.parse_args()
    config_path = args.params_path
    # Read config
    config = src.config.UpstreamConfig(config_path)
    ds_path = args.dataset_path
    upstream(config, ds_path)
