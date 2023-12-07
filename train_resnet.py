## Silvija Kokalj-Filipovic
## Final exam for the Deep Learning class
## November 2023

#run_resnetI is for training the resnet classifier

#test_recon_model is for evaluationg reconstructions on the resnet classifier


from utils import *
#from resnet_I_18 import resnetI18
from resnet_I_50 import resnetI50
from pytorch_resnetI import test_model
import matplotlib.pyplot as plt
from load_datasets import load_ImageNet100
from torch.utils.data import DataLoader
import os
from evaluate_recons import evaluate_dataset
from torchattacks import FGSM
import torchvision.models as models
import torch.nn as nn
from torchinfo import summary


def save_training_metrics(train_losses, valid_losses, save_visual_name): 
    '''
    Args: 
    - model: Lenet Model instance.
    - dl_test: DataLoader containing test data.
    - train_losses: list of training loss values from training.
    - valid_losses: list of validation loss values from training. Could be empty.
    - save_visual_name: file name for the training and validation loss graph to be saved as.
    - model_name: name of the model. To be input into accuracies csv.
    - ds_name: name of the dataset. To be input into accuracies csv.
    '''

    if not os.path.isdir(MISC_VIS_DIR):
        os.makedirs(MISC_VIS_DIR)

    plt.plot(range(1, len(train_losses) + 1), train_losses, label = "Training Loss")
    if len(valid_losses) > 0:
        plt.plot(range(1, len(valid_losses) + 1), valid_losses, label = "Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("Training Loss values")
    plt.legend()
    plt.savefig(os.path.join(MISC_VIS_DIR, save_visual_name))
    plt.clf()

def run_resnetI(dl_train, dl_valid, save_path, save_visual_name, num_classes, stop_early = False, validate = False, attack = None):
    """
    Args: 
    - dl_train: data loader instance with our training information.
    - dl_valid: data loader instance with our validation information.
    - dl_test: data loader instance with our testing information.
    - save_path: file path to save the model to.
    - save_visual_name: file name for the training loss visual to be made.
    - ds_name: the name of the dataset.
    - num_classes: how many classes are in the dataset
    - stop_early: should the model stop training early if validation loss increases.
    - validate: should the model perform validation after each epoch
    """

    model = resnetI50(dl_train, dl_valid, num_classes, save_path, stop_early)
    train_losses, valid_losses = model.run_epochs(n_epochs= 10, validate=validate, attack=attack)
    save_training_metrics(train_losses, valid_losses, save_visual_name)
    return model

def test_recon_model(model, ds_test, ds_recon_test, ds_name, attack = None): 
    model.eval()
 
    dl_test = DataLoader(ds_test, batch_size=IMAGENET100_BATCH_SIZE, num_workers=NUM_DATA_LOADER_WORKERS, shuffle = False)
    dl_recon_test = DataLoader(ds_recon_test, batch_size=IMAGENET100_BATCH_SIZE, num_workers=NUM_DATA_LOADER_WORKERS, shuffle = False)

    predictions, labels = test_model(model, dl_test)
    correct_idxs, incorrect_idxs = evaluate_dataset('HAE', labels, predictions, ds_name, attack = "None")
    rec_predictions, _ = test_model(model, dl_recon_test)
    correct_rec_idxs, incorrect_rec_idxs = evaluate_dataset('HAE', labels, rec_predictions, ds_name,  attack = "None")

    if attack is not None: 
        atk_predictions, labels = test_model(model, dl_test, attack)
        correct_atc_idxs, incorrect_atc_idxs  = evaluate_dataset('HAE', labels, atk_predictions, ds_name, attack = attack.attack)
    else:
        correct_atc_idxs=1
        incorrect_atc_idxs =1
    return correct_idxs, incorrect_idxs, correct_rec_idxs, incorrect_rec_idxs, correct_atc_idxs, incorrect_atc_idxs

if __name__ == "__main__":

    # Regular ImageNet100
    dl_train, dl_valid, _ = load_ImageNet100(validate=True)
    resnet_model = run_resnetI(dl_train, 
            dl_valid, 
            RESNET50_I100_PATH, 
            "resnet50_I100.png", 
            100,
            validate = True)

    del resnet_model


