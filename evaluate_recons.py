import numpy as np
from utils import *
from torch.utils.data import DataLoader, Subset
import os
import torch
import torchattacks
from tqdm import tqdm
from pytorch_resnetI import test_model, outputs_to_predictions
#from tsne import run_tsne
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
 
from reconstruct_avg_attack import run_reconstruct_avg, run_dict_reconstruct_avg

def evaluate_dataset(model_name, test_labels, predictions, ds_name, save_result = True, attack = None):
    """
    Arguments: 
    - model_name : name of the model that gave the predictions.
    - test_labels : list ground truth labels of the images. 
    - predictions : list of predicted labels made by the model. 
    - ds_name : name of the dataset that is being evaluated.
    - recon_name : the name of the reconstruction layer. 
    - ret_correct_idxs : should the method return the indicies of the images predicted correctly?
    - save_result: should this evaluation be saved to the classification_accuracies.csv file?
    - attack: the name of the attack that has been applied to the model (if any). """

    correct_idxs = []
    incorrect_idxs = []
    
    for i, (pred, label) in enumerate(zip(predictions, test_labels)):
        if label == pred:
            correct_idxs.append(i)
        elif label != pred:
            incorrect_idxs.append(i)

    '''if save_result:
        if not os.path.isdir(CONF_MAT_VIS_DIR): 
            os.mkdir(CONF_MAT_VIS_DIR)
        conf_mat_disp = ConfusionMatrixDisplay.from_predictions(test_labels, predictions, normalize = "all")
        conf_mat_disp = conf_mat_disp.plot()
        plt.title(f"{model_name} Confusion Matrix on {ds_name}: {recon_name}")
        plt.savefig(os.path.join(CONF_MAT_VIS_DIR, f"{model_name}_{ds_name}_{recon_name}.png"))
        plt.close("all")

        avg_accuracy = len(correct_idxs) / len(test_labels)
        attack_name = attack if attack != None else "None"
        add_accuracy_results(model_name, ds_name, recon_name, attack_name, avg_accuracy)'''

    return correct_idxs, incorrect_idxs


def sample_by_class(ds_test, correct_idxs, correct_predictions, incorrect_idxs, incorrect_predictions, force_even = False): 

    classes = np.unique(correct_predictions)

    inc_count_map = {num: 0 for num in classes}
    cor_img_map = {num: [] for num in classes}
    return_img_map = {num: [] for num in classes}

    for idx in incorrect_idxs: 
        inc_count_map[incorrect_predictions[idx]] += 1
    
    for idx in correct_idxs: 
        cor_img_map[correct_predictions[idx]].append(ds_test[idx][0])

    for num, count in inc_count_map.items():
        if force_even and count % 2 == 1: 
            count += 1
        return_img_map[num].extend(random.sample(cor_img_map[num], count))

    return return_img_map

