from torchvision.datasets import VisionDataset
from PIL import Image
from torchvision.datasets.utils import download_url

import opendatasets as od
import tarfile

import os
import os.path
from tqdm import tqdm
import sys
import numpy as np



#https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
DATASETS = {'101': {'file_id': '137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp', 'url': 'https://drive.google.com/u/0/uc?id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp&export=download'},
        '256': {'file_id': '1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK', 'url': 'https://drive.google.com/u/0/uc?id=1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK&export=download'}}

CHUNK_SIZE = 32768


def download_caltech(path, type='101'):
    '''
    Download the archive file of Caltech dataset

    :param path: str, dir-path
                Path where to download the archive file

    :param type: str
                Type of Caltech dataset. It can be '101' or '256
    '''

    import requests

    dataset = DATASETS.get(type)
    if dataset is None:
        raise Exception("Dataset type not valide. Choose type '101' or type '256'")

    # Set connection
    dataset_url = dataset['url']
    dataset_file_id = dataset['file_id']
    session = requests.Session()
    response = session.get(dataset_url, params = {'id': dataset_file_id})

    # retrieve token
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token =  value
            params = {'id': dataset_file_id, 'confirm': token}
            response = session.get(dataset_url, params=params, stream=True)
            break

    file_path = path+f"/Caltech-{type}.tar"

    # Download file
    print("* Downloading file")
    with open(file_path, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def unzip_caltech(file_path, destination):
    print("* Unzipping compressed file")
    with tarfile.open(file_path) as tar:
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
            tar.extract(member=member, path=destination)


def split_caltech(root, train_file = './train.txt', test_file = './test.txt', test_size=0.2):
    '''
    Produce two files defining the train/test split of a Caltech dataset.
    Each file contains a list of relative paths to the images in the dataset.
    Each image is assigned either to train or test set

    > train.txt
    .\101_ObjectCategories\accordion\image_0001.jpg
    .\101_ObjectCategories\accordion\image_0005.jpg
    ...
    '''

    from glob import glob

    print("* Preparing train/test split")
    with open(train_file, 'w') as train_file, open(test_file, 'w') as test_file:
        for file_name in tqdm(glob(root + "/*/*")):
            if np.random.rand() > test_size:
                train_file.write(file_name+'\n')
            else:
                test_file.write(file_name + '\n')




def download_and_prepare_caltech(path, type='101', test_size=0.2):
    f'''
    The function will download a tar file containing the specified Caltech dataset, extracting it and preparing two files
    containing the train/test files split
    
    :param path: str, dir-path 
            path where the archive will be extracted. If downloading Caltech-101, the root of the dataset will be at
            path +'/101-ObjectCategories'
    
    :param type: str
            type of the dataset. It can be '101' or '256'
            
    :param test_size: float in range [0,1] 
            define the percentage of train/test set in the split
    '''

    # download_caltech(path, type)
    file_path = path + f"/Caltech-{type}.tar" + ('.gz' if type == '256' else '')
    unzip_caltech(file_path, path)

    os.remove(file_path)

    split_caltech(os.path.join(path, f"{type}_ObjectCategories"),
                  train_file=f'./{type}-train.txt',
                  test_file=f'./{type}-test.txt',
                  test_size=test_size)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


if __name__ == '__main__':
    download_and_prepare_caltech('.', type='101')