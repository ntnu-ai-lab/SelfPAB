import os
import h5py
import argparse
import torch
from tqdm import tqdm
import src.config, src.utils, src.datasets, src.models


def extract_features(config, ds_path, batch_size=1, subsample_perc=1.0):
    '''Creates embeddings of the dataset using the given upstream model

    This code is not part of the main pipeline but useful when one is
    interested in the upstream model's generated feature embeddings

    '''
    ds_args = src.utils.grid_search(config.DATASET_ARGS)[0]
    args = src.utils.grid_search(config.ALGORITHM_ARGS)[0]
    dataset = src.datasets.get_dataset(
        dataset_name=config.DATASET,
        dataset_args=ds_args,
        root_dir=ds_path,
        num_classes=config.num_classes,
        label_map=config.label_index,
        replace_classes=config.replace_classes,
        config_path=config.CONFIG_PATH,
        #test_mode=True, inference_mode=True
    )
    if subsample_perc < 1.0:
        print('############# Subsample data with: ', subsample_perc)
        sampler = torch.utils.data.RandomSampler(
            data_source=dataset,
            num_samples=int(len(dataset)*subsample_perc)
        )
    else:
        sampler = None
    dl = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config.NUM_WORKERS
    )
    all_labels_in_dl = []
    for i in dl:
        all_labels_in_dl.append(torch.unique(torch.flatten(i[1])))
    all_labels_in_dl = torch.unique(torch.concat(all_labels_in_dl))
    if len(all_labels_in_dl) != config.num_classes:
        raise Exception('Not all labels in train_ds')
    args.update({'input_dim': dataset.feature_dim,
                 'output_dim': dataset.output_shapes,
                 'total_step_count': len(dl)})
    model = src.models.get_model(
        algorithm_name=config.ALGORITHM,
        algorithm_args=args
    )
    model.eval()
    features_path = os.path.join(config.CONFIG_PATH, 'feature_emb')
    if not os.path.exists(features_path):
        os.makedirs(features_path)
    f_path = os.path.join(features_path, 'features.h5')
    l_path = os.path.join(features_path, 'labels.h5')

    # Iterate through the DataLoader
    bidx = 0
    with h5py.File(f_path, 'w') as f, \
            h5py.File(l_path, 'w') as l:
        for batch in tqdm(dl):
            with torch.no_grad():
                features = model.upstream_models[0](batch[0])
            labels = batch[1]
            features = features.view(-1, features.shape[-1])
            labels = labels.view(-1)

            f.create_dataset(f'batch_{bidx}', data=features.numpy())
            l.create_dataset(f'batch_{bidx}', data=labels.numpy())
            bidx+=1
    print(f'Done. Stored features at: {features_path}')


parser = argparse.ArgumentParser(
    description='Extracts features from the pre-trained model.'
)
parser.add_argument(
    '-p', '--params_path', required=False, type=str,
    help='params path with config.yml file',
    default='params/selfPAB_downstream_experiments/selfPAB_downstream_harth/config.yml'
)
parser.add_argument(
    '-d', '--dataset_path', required=False, type=str,
    help='path to dataset.',
    default='data/harth/'
)
parser.add_argument(
    '--subsample_perc', required=False, type=float,
    help='How much in percent of the dataset to use (0-1]',
    default=1.0
)
args = parser.parse_args()
config_path = args.params_path
# Read config
config = src.config.Config(config_path)
ds_path = args.dataset_path
print('Extract features and corresponding labels')
extract_features(
    config=config,
    ds_path=ds_path,
    batch_size=256,
    subsample_perc=args.subsample_perc
)
