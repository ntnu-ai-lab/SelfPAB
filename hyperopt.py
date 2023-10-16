import argparse
import loo_cross_validation
import src.config
from ray import tune
from functools import partial
import os


def hyperopt(config, dataset_path=None,
             num_trials=None, name=None, resume=False,
             gpu_per_trial=1):
    '''Starts hyperopt based on parameters in config'''
    print('GPU per trial: ', gpu_per_trial)
    if resume == 'False':
        resume = False
    elif resume == 'True':
        resume = True
    config.WANDB = False  # Avoid logging in WANDB for this
    config.FOLDS = 1  # Single random split during hyperopt (final 5-fold/LOSO on best hyperparams)
    config.VALID_SPLIT = 0.0  # During hyperopt we focus on test performance
    _num_trials = prepare_config(config)
    num_trials = num_trials if num_trials else _num_trials
    os.environ['GRPC_ENABLE_FORK_SUPPORT'] = "false"  # Avoid print spam
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(g) for g in config.NUM_GPUS])
    config.NUM_GPUS = [0]
    try:
        result = tune.run(
            partial(loo_cross_validation.loso_cv, dataset_path=dataset_path, hopt=True),
            resources_per_trial={"cpu": config.NUM_WORKERS, "gpu": gpu_per_trial},
            num_samples=num_trials,
            config=config,
            checkpoint_at_end=False,
            mode='max',
            name=name,
            resume=resume
        )
    except ray.tune.error.TuneError as e:
        print(e)
    best_trial = result.get_best_trial("score_mean", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    best_score_mean = best_trial.last_result["score_mean"]
    best_score_std = best_trial.last_result["score_std"]
    print(f'Best trial final {config.EVAL_METRIC}: {best_score_mean}({best_score_std})')


def prepare_config(config):
    '''Adapts config to be usable in hyperopt, gives resulting num trials'''
    default_trials = 1
    for hpar_name, hpar_vals in config['ALGORITHM_ARGS'].items():
        if len(hpar_vals)<=1: continue
        config['ALGORITHM_ARGS'][hpar_name] = tune.grid_search(hpar_vals)
        default_trials *= len(hpar_vals)
    for hpar_name, hpar_vals in config['DATASET_ARGS'].items():
        if len(hpar_vals)<=1: continue
        config['DATASET_ARGS'][hpar_name] = tune.grid_search(hpar_vals)
        default_trials *= len(hpar_vals)
    #return default_trials
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start hyperopt based on CV.')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='/param/config.yml')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default=None)
    parser.add_argument('-n', '--num_trials', required=False, type=int,
                        help='how many trials.', default=None)
    parser.add_argument('-g', '--gpu_per_trial',
                        required=False, type=float,
                        help='how much GPU power to use for 1 trial',
                        default=1.0)
    parser.add_argument('--name', required=False, type=str,
                        help='Name of run.', default=None)
    parser.add_argument(
        '--resume',
        required=False,
        type=str,
        help='Use "LOCAL+RESTART_ERRORED" for restoring <name>',
        default="False"
    )
    args = parser.parse_args()
    config_path = args.params_path
    # Read config
    config = src.config.Config(config_path)
    ds_path = args.dataset_path
    hyperopt(config, ds_path, args.num_trials, name=args.name, resume=args.resume, gpu_per_trial=args.gpu_per_trial)
