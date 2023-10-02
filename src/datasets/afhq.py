from typing import Any, Tuple
import torch
from torchvision.datasets import ImageFolder
import sys, os
import pathlib
import shutil
import subprocess
from torchvision.transforms import ToTensor, Compose, Normalize

file_dir = pathlib.Path(__file__).parent.resolve()

DATASET_PATH = os.path.join(file_dir, 'data/afhq')
# print(DATASET_PATH)

class ImageFolderNoClass(ImageFolder):

    def __getitem__(self, index: int) -> Any:
        ims, clss =  super().__getitem__(index)
        return ims

def check_image_folder(path):
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if len(onlyfiles) == 0:
        return False
    for file in onlyfiles:
        img_extensions = ['.jpg', '.png']
        for im_ext in img_extensions:
            if file.endswith(im_ext):
                return True
    raise Exception('Strange folder!')

def get_afhq_dataset(dstype: str, train=True, transform=None, pix_range=(-1, 1.)):
    assert dstype in ['cat', 'dog', 'wild']
    _interm = 'train' if train else 'val'
    ULTIMATE_PATH = os.path.join(DATASET_PATH, _interm, dstype)
    if check_image_folder(ULTIMATE_PATH):
        new_ulti_path = os.path.join(ULTIMATE_PATH, 'dummy_class')
        os.makedirs(new_ulti_path)
        mv_command = "mv {} {}".format(os.path.join(ULTIMATE_PATH, '*.jpg'), new_ulti_path)
        subprocess.call(mv_command, shell=True)
        # shutil.move(os.path.join(ULTIMATE_PATH, '*.jpg'), new_ulti_path)
    
    _mean, _std = - pix_range[0] / (pix_range[1] - pix_range[0]), 1./(pix_range[1] - pix_range[0])
    basic_transform = Compose([ToTensor(), Normalize(_mean, _std)])
    if transform is None:
        transform = basic_transform
    else:
        transform = Compose([transform, basic_transform])
    return ImageFolderNoClass(ULTIMATE_PATH, transform=transform) # TODO: CREATE DUMMY CLASS FOLDER IF DATASET_PATH CONSIST OF IMAGES!!!

if __name__ == '__main__':
    ds = get_afhq_dataset('cat', train=False)
    print(len(ds))
