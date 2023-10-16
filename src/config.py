import os
import yaml
import functools


def cached(f):
    """
    Decorator that creates a cached property.
    """
    key = f.__name__

    @property
    @functools.wraps(f)
    def decorated(self):
        if key not in self._cache:
            self._cache[key] = f(self)
        return self._cache[key]

    return decorated


class Config(dict):
    def __init__(self, path):
        self._cache = {}
        if type(path) == dict:
            cfg = path
        else:
            with open(path, 'r') as f:
                cfg = yaml.load(f, Loader=yaml.Loader)
        self.TRAIN_DATA = cfg['TRAIN_DATA']
        self.CLASSES = cfg['CLASSES']
        self.VALID_SPLIT = cfg['VALID_SPLIT']
        self.TEST_SPLIT = cfg['TEST_SPLIT'] if 'TEST_SPLIT' in cfg else None
        self.FOLDS = cfg['FOLDS'] if 'FOLDS' in cfg else 0
        self.EARLY_STOPPING = cfg['EARLY_STOPPING'] if 'EARLY_STOPPING' in cfg else False
        self.TEST_SUBJECTS = cfg['TEST_SUBJECTS']
        self.EVAL_METRIC = cfg['EVAL_METRIC']
        self.STORE_CMATS = cfg['STORE_CMATS']
        self.SKIP_FINISHED_ARGS = cfg['SKIP_FINISHED_ARGS']
        self.SEED = cfg['SEED']
        self.NUM_WORKERS = cfg['NUM_WORKERS']
        self.NUM_GPUS = cfg['NUM_GPUS']
        self.ALGORITHM = cfg['ALGORITHM']
        self.ALGORITHM_ARGS = cfg['ALGORITHM_ARGS']
        self.DATASET = cfg['DATASET']
        self.DATASET_ARGS = cfg['DATASET_ARGS']
        self.PROJ_NAME = cfg['PROJ_NAME'] if 'PROJ_NAME' in cfg else ''
        self.WANDB = cfg['WANDB']
        self.WANDB_KEY = cfg['WANDB_KEY'] if 'WANDB_KEY' in cfg else ''
        self.ADDITIONAL_EVAL_METRICS = cfg['ADDITIONAL_EVAL_METRICS'] if \
                'ADDITIONAL_EVAL_METRICS' in cfg else []
        self.METRIC_AGGR_WINDOW_LEN = cfg['METRIC_AGGR_WINDOW_LEN'] if \
                'METRIC_AGGR_WINDOW_LEN' in cfg else None
        # Path of this config file
        try:
            self.CONFIG_PATH = os.path.dirname(os.path.realpath(path))
        except TypeError:
            self.CONFIG_PATH = cfg['CONFIG_PATH']
        param_dict = self.__dict__.copy()
        param_dict.pop('_cache')
        super(Config, self).__init__(param_dict)

    @property
    def all_classes(self):
        """
        Get all classes if present.
        """
        if self.CLASSES is None:
            raise Exception('Config file does not have "CLASSES" member')
        return self.CLASSES

    @property
    def classes(self):
        """
        Get all non-replaced classes.
        """
        return [c for c in self.all_classes if not 'replace' in c]

    @property
    def replace_classes(self):
        """
        Get replace dict for classes that have been dropped.
        """
        return {c['label']: c.get('replace', c['label']) for c in self.all_classes}

    @cached
    def num_classes(self):
        return len(self.classes)

    @property
    def label_index(self):
        '''Index of each label defined in CLASSES'''
        return {c['label']: i for i, c in enumerate(self.classes)}

    @property
    def class_names(self):
        """
        Get all non-replaced class names.
        """
        return [c['name'] for c in self.classes]

    @property
    def possible_indices(self):
        '''
        Possible label indices based on classes
        '''
        return list(self.label_index.values())

    @property
    def class_label_name_map(self):
        """
        Get non-replaced class value to class name dict
        """
        return {x['label']: x['name'] for x in self.classes}


class UpstreamConfig:
    def __init__(self, path):
        self._cache = {}
        with open(path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        self.TRAIN_DATA = cfg['TRAIN_DATA']
        self.SEED = cfg['SEED']
        self.NUM_WORKERS = cfg['NUM_WORKERS']
        self.NUM_GPUS = cfg['NUM_GPUS']
        self.ALGORITHM = cfg['ALGORITHM']
        self.ALGORITHM_ARGS = cfg['ALGORITHM_ARGS']
        self.DATASET = cfg['DATASET']
        self.DATASET_ARGS = cfg['DATASET_ARGS']
        self.WANDB = cfg['WANDB']
        # Path of this config file
        self.CONFIG_PATH = os.path.dirname(os.path.realpath(path))
        self.PROJ_NAME = cfg['PROJ_NAME'] if 'PROJ_NAME' in cfg else ''
