import os
import argparse
import subprocess
import tempfile
import zipfile


def download_harth():
    link = 'https://archive.ics.uci.edu/static/public/779/harth.zip'
    with tempfile.TemporaryDirectory() as tmpdirname:
        print('Downloading HARTH dataset...')
        runcmd(f'wget -P {tmpdirname} {link}', verbose=False)
        zip_file = os.listdir(tmpdirname)[0]
        with zipfile.ZipFile(tmpdirname+'/'+zip_file, 'r') as zip_ref:
            zip_ref.extractall('data/')
    print('Done. Dataset in: data/harth/')


def download_har70plus():
    link = 'https://archive.ics.uci.edu/static/public/780/har70.zip'
    with tempfile.TemporaryDirectory() as tmpdirname:
        print('Downloading HAR70+ dataset...')
        runcmd(f'wget -P {tmpdirname} {link}', verbose=False)
        zip_file = os.listdir(tmpdirname)[0]
        with zipfile.ZipFile(tmpdirname+'/'+zip_file, 'r') as zip_ref:
            zip_ref.extractall('data/')
    print('Done. Dataset in: data/har70plus/')


def runcmd(cmd, verbose=False, *args, **kwargs):
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


if __name__ == '__main__':
    allowed_datasets = ['harth', 'har70plus']
    parser = argparse.ArgumentParser(
        description=f'Download and process datasets: {allowed_datasets}'
    )
    parser.add_argument(
        'dataset_name',
        type=str,
        help=f'Dataset name to download. Allowed: {allowed_datasets}'
    )
    args = parser.parse_args()
    ds_name = args.dataset_name
    if ds_name == 'harth':
        download_harth()
    elif ds_name == 'har70plus':
        download_har70plus()
    else:
        raise ValueError(f'Unsuported dataset name: {ds_name}')
