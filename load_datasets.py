import lmdb
import pickle
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from torchsig.datasets.modulations import ModulationsDataset
from torchsig.utils.dataset import SignalDataset
import torchsig.transforms as ST
from torchsig.utils.visualize import IQVisualizer, SpectrogramVisualizer

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split
from utils import *
from Datasets.ImageNetDataset import ImageNet100
from torch.utils.data import random_split


def _make_train_valid_split(ds_train, len_ds_test):
    train_idxs, valid_idxs, _, _ = train_test_split(
            range(len(ds_train)),
            ds_train.targets,
            #stratify=ds_train.targets,
            test_size= len_ds_test / len(ds_train), 
            random_state=RANDOM_SEED
        )
    ds_train = Subset(ds_train, train_idxs)
    ds_valid = Subset(ds_train, valid_idxs)
    
    return ds_train, ds_valid

def _make_data_loaders(ds_train, ds_test, validate, num_workers=4, batch_size=IMAGENET100_BATCH_SIZE):
    ds_valid = None
    if validate:
        ds_train, ds_valid = random_split(ds_train, [len(ds_train) - 8*len(ds_test),8*len(ds_test)])
    #import ipdb;ipdb.set_trace()
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    dl_valid = None
    if ds_valid:
        dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)

    
    return dl_train, dl_valid, dl_test

def load_ImageNet100(validate = False):
    ds_train = ImageNet100(DATA_FOLDER, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))
    ds_test = ImageNet100(DATA_FOLDER, split="val", transform=transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]))

    return _make_data_loaders(ds_train, ds_test, validate)


