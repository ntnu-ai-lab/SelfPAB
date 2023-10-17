# SelfPAB
Implementation of the SelfPAB method presented in our paper: Large-Scale Pre-Training for Dual-Accelerometer Human Activity Recognition.

## Requirements
- Python 3.8.10
```bash
pip install -r requirements.txt
```

## Download Datasets
Download the required datasets. Currently the downstream datasets [HARTH](https://archive.ics.uci.edu/dataset/779/harth) and [HAR70+](https://archive.ics.uci.edu/dataset/780/har70) are supported. The HUNT4 subset used for pre-training is planned to be published in future releases.
```bash
python download_dataset.py <dataset_name>
# Example: python download_dataset.py harth
```
This command will download the given dataset into the [data/](https://github.com/ntnu-ai-lab/SelfPAB/tree/main/data) folder.

## Upstream Pre-training
The upstream training can be started with:
```bash
python upstream.py -p <path/to/config.yml> -d <path/to/dataset>
# Example (SelfPAB): python upstream.py -p params/selfPAB_upstream/config.yml -d data/hunt4/
```
The [params/selfPAB_upstream/config.yml](https://github.com/ntnu-ai-lab/SelfPAB/blob/main/params/selfPAB_upstream/config.yml) contains all upstream model hyperparameters, presented in our paper.

After upstream pre-training, the model with the best validation loss is stored as .ckpt file. This saved model can now be used as feature extractor for downstream training. The [parmas/selfPAB_upstream/upstream_model.ckpt](https://github.com/ntnu-ai-lab/SelfPAB/blob/main/params/selfPAB_upstream/upstream_model.ckpt) is the transformer encoder, which we pre-trained on 100,000 hours of unlabeled HUNT4 data using the masked reconstruction auxiliary task, as presented in our paper: Large-Scale Pre-Training for Dual-Accelerometer Human Activity Recognition.

## Downstream Training
A downstream leave-one-subject-out cross-validation (LOSO) can be started with:
```bash
python loo_cross_validation.py -p <path/to/config.yml> -d <path/to/dataset>
# Example (SelfPAB): python loo_cross_validation.py -p params/selfPAB_downstream_harth/config.yml -d data/harth/
```
The downstream config has an argument called "upstream_model_path", defining the upstream model to use. In the [above example](https://github.com/ntnu-ai-lab/SelfPAB/blob/main/params/selfPAB_downstream_harth/config.yml) the upstream model is set to be our pre-trained transformer encoder: [parmas/selfPAB_upstream/upstream_model.ckpt](https://github.com/ntnu-ai-lab/SelfPAB/blob/main/params/selfPAB_upstream/upstream_model.ckpt).

## Results Visualization
The training results/progress can be logged in [Weights and Biases](https://wandb.ai/) by setting the WANDB argument in the config.yml files to True. For downstream training, the WANDB_KEY has to be provided in the config.yml file as well.

## Citation
If you use the [pre-trained SelfPAB upstream model](https://github.com/ntnu-ai-lab/SelfPAB/blob/main/params/selfPAB_upstream/upstream_model.ckpt) for your research, please cite the following paper:
