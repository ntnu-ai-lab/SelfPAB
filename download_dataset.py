import os
import argparse
import subprocess
import tempfile
import zipfile
import shutil


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

def download_pamap2(only_protocol_data=True):
    link = 'https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip'
    trgt_dir = 'data/pamap2/'
    with tempfile.TemporaryDirectory() as tmpdirname:
        if only_protocol_data:
            print('Downloading PAMAP2 Protocol data...')
        else:
            print('Downloading PAMAP2 dataset...')
        runcmd(f'wget -P {tmpdirname} {link}', verbose=False)
        zip_file = os.listdir(tmpdirname)[0]
        with zipfile.ZipFile(tmpdirname+'/'+zip_file, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as tmpsubdirname:
                zip_ref.extractall(tmpsubdirname)
                zip_sub_path = tmpsubdirname+'/PAMAP2_Dataset.zip'
                with zipfile.ZipFile(zip_sub_path, 'r') as zip_sub_ref:
                    if only_protocol_data:
                        zip_sub_ref.extractall(tmpsubdirname)
                        shutil.move(tmpsubdirname+'/PAMAP2_Dataset/Protocol',
                                    trgt_dir)
                    else:
                        zip_sub_ref.extractall(trgt_dir)
    print(f'Done. Dataset in: {trgt_dir}')

def download_uschad(only_subjects=True):
    link = 'https://sipi.usc.edu/had/USC-HAD.zip'
    trgt_dir = 'data/uschad/'
    if not os.path.exists(trgt_dir):
        os.makedirs(trgt_dir)
    with tempfile.TemporaryDirectory() as tmpdirname:
        if only_subjects:
            print('Downloading USC-HAD Subject data...')
        else:
            print('Downloading USC-HAD dataset...')
        runcmd(f'wget -P {tmpdirname} {link}', verbose=False)
        zip_file = os.listdir(tmpdirname)[0]
        with zipfile.ZipFile(tmpdirname+'/'+zip_file, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as tmpsubdirname:
                zip_ref.extractall(tmpsubdirname)
                source_folder = tmpsubdirname+'/USC-HAD/'
                files_to_move = [p for p in os.listdir(source_folder)]
                if only_subjects:
                    files_to_move = [p for p in files_to_move if 'Subject' in p]
                for f in files_to_move:
                    shutil.move(source_folder+f, trgt_dir)
    print(f'Done. Dataset in: {trgt_dir}')


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
    allowed_datasets = ['harth', 'har70plus', 'pamap2', 'mobiact', 'uschad']
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
    elif ds_name == 'pamap2':
        download_pamap2()
    elif ds_name == 'uschad':
        download_uschad()
    elif ds_name == 'mobiact':
        print('Warning: To get acces to the MobiAct dataset, please contact: bmi@hmu.gr \n\t For more details see: https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/')
    else:
        raise ValueError(f'Unsuported dataset name: {ds_name}')
