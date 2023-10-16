import os
import argparse
import train
import src.config
import src.datasets
import torch
import numpy as np
import random
from ray import tune

def loso_cv(config, dataset_path=None, hopt=False):
    '''Starts a leave-one-out cross validation'''
    # Set all seeds:
    if type(config) == dict:
        config = src.config.Config(config)
        if hopt:
            config.WANDB = False
            config.FOLDS = 1
            config.VALID_SPLIT = 0.0
            config.NUM_GPUS = [0]
    print(type(config))
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # In LOSO test subject is used for valid
    valid_split = 'test' if config.FOLDS==0 else config.VALID_SPLIT
    print('valid_split=',valid_split)
    if config.WANDB:
        ds_name = os.path.realpath(dataset_path).split('/')[-1]
        proj_name = 'harth_plus_dl_LOSO_'+config.PROJ_NAME+ds_name
        run_name = config.ALGORITHM+'_'+config.DATASET
        src.utils.wandb_init(
            run_name=run_name,
            wandb_config=vars(config),
            entity='hunt4-har',
            proj_name=proj_name,
            key=config.WANDB_KEY
        )
        all_test_true = []
        all_test_pred = []
    filenames = sorted(src.utils.get_filenames_of_dataset(dataset_path))
    if len(filenames) == 0:
        filenames = sorted(src.utils.get_filenames_of_dataset(dataset_path,
                                                              filetype='dat'))
    fold_performances = []
    for fold, train_files, test_files in src.utils.cv_split(filenames,
                                                            config.FOLDS,
                                                            config.SEED,
                                                            config.TEST_SPLIT):
        if type(valid_split)==float:
            config.VALID_SPLIT = random.sample(
                train_files,
                int(valid_split * len(train_files))
            )
        else:
            config.VALID_SPLIT = valid_split
        config.TEST_SUBJECTS = test_files
        print(f'Test subject: {test_files}')
        print(f'Train subjects: {train_files}')
        _,test_cmat,best_logs,best_args = train.train(config,dataset_path,loso=True)
        if config.WANDB:
            for test_filename in test_cmat.keys():
                src.utils.log_cmat_metrics_to_wandb(
                    log_cmat=test_cmat[test_filename],
                    log_name=test_filename,
                    class_names=config.class_names,
                    metrics=['average_f1score',
                             'average_recall',
                             'average_precision',
                             'accuracy',
                             'cmat',
                            ]
                )
                src.utils.log_history_metrics_to_wandb(
                    metrics_dict=best_logs,
                    log_name=test_filename,
                )
                all_test_true += list(test_cmat[test_filename].y_true)
                all_test_pred += list(test_cmat[test_filename].y_pred)
        fold_performances.append(np.mean([getattr(_c, config.EVAL_METRIC) for _c in test_cmat.values()]))
        if config.STORE_CMATS:
            to_store_path = f'{config.CONFIG_PATH}/loso_cmats/' if config.FOLDS==0 \
                       else f'{config.CONFIG_PATH}/CV_folds{config.FOLDS}_cmats/'
            if test_cmat is None:
                breakpoint()
            for test_filename in test_cmat.keys():
                to_store_filename = test_filename.split('.')[-2] if '.' in test_filename \
                                    else test_filename
                src.utils.save_intermediate_cmat(
                    path=to_store_path,
                    filename=to_store_filename + '_cmat.pkl',
                    args=best_args,
                    cmats={test_filename: test_cmat[test_filename]},
                    valid_subjects=[test_filename]
                )
    final_CV_test_perf_mean = np.mean(fold_performances)
    final_CV_test_perf_std = np.std(fold_performances)
    print(f'Final {config.FOLDS}-fold CV {config.EVAL_METRIC}: {final_CV_test_perf_mean}({final_CV_test_perf_std})')
    if hopt:
        tune.report(score_mean=final_CV_test_perf_mean,
                    score_std=final_CV_test_perf_std,
                    score_name=config.EVAL_METRIC)

    if config.WANDB:
        src.utils.log_wandb_cmat(
            y_true=all_test_true,
            y_pred=all_test_pred,
            class_names=config.class_names,
            log_name='Total'
        )




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start LOSO CV.')
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
    loso_cv(config, ds_path)
