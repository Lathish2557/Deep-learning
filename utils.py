import torch
import random
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from PIL import Image
from pathlib import Path
from torchvision.utils import make_grid
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import torchvision.transforms.functional as TF
import scipy.fftpack as fp

device = "cuda:1" if torch.cuda.is_available() else "cpu"
FGSM_EPS = 16 / 255

DATA_FOLDER = '/home/dl_class/data/ILSVRC/Data/CLS-LOC/'

LAYER_NAMES = ["Layer 0", "Layer 1", "Layer 2", "Layer 3", "Layer 4 Final"]
RECON_ROOT_NAMES = ["data_original", "data_jpg", "data_recon_0", "data_recon_1",
                    "data_recon_2", "data_recon_3", "data_recon_4"]


CWD = os.path.abspath(os.getcwd())
JSON_DIR_PATH =os.path.join(CWD, "json_files")

#Saved data file paths (original data and constructions)
IMG_DIR_PATH = os.path.join(CWD, "data")

#Model relatd file paths
MODELS_DIR = os.path.join(CWD, "models")
LOG_DIR = os.path.join(MODELS_DIR, "log")
CONF_MAT_VIS_DIR = os.path.join(MODELS_DIR, "confusion")



RESNET18_I100_PATH = os.path.join(MODELS_DIR, "resnet18_imagenet100.pt")
RESNET50_I100_PATH = os.path.join(MODELS_DIR, "resnet50_imagenet100.pt")


# Visualization Directories
VISUAL_DIR = os.path.join(CWD, "Visuals")
MISC_VIS_DIR = os.path.join(VISUAL_DIR, "Miscellaneous")
ACCURACY_VIS_DIR = os.path.join(VISUAL_DIR, "Accuracy Plots")

RECONS_EXPERIMENT_DIR = os.path.join(CWD, "Recons_Experiment")

ACCURACY_OUTPUT_FILE = os.path.join(CWD, "classification_accuracies.csv")
ACCURACY_FILE_COLS = ["Model", "Dataset", "Reconstruction", "Attack", "Average Accuracy"] 
IMAGENET100_BATCH_SIZE = 16
NUM_DATA_LOADER_WORKERS = 4
RANDOM_SEED = 42






def hflip_image(img): 
    return TF.hflip(img)


def rotate_image(img):
    return TF.rotate(img, -90)




def add_row_to_csv(output_file, output_cols, row_dict):
    results_df = pd.read_csv(output_file, index_col=False) \
                if os.path.exists(output_file) \
                else pd.DataFrame(columns = output_cols) 

    row_df = pd.DataFrame(row_dict, columns = output_cols)
    results_df = pd.concat([results_df, row_df], ignore_index=True)
    results_df.to_csv(output_file, index = False)

def add_accuracy_results(model_name, dataset_name, reconstruction, attack_name, avg_accuracy):

    row_dict = {"Model" : [model_name], 
                "Dataset" : [dataset_name], 
                "Reconstruction" : [reconstruction],
                "Attack" : [attack_name],
                "Average Accuracy" : [avg_accuracy]}

    add_row_to_csv(ACCURACY_OUTPUT_FILE, ACCURACY_FILE_COLS, row_dict)



CLASSIFICATION_IDXS_FILE = os.path.join(CWD, "classification_idxs.csv")
CLASSIFICATION_IDXS_COLS = ["Model", "Dataset", "Reconstruction", "Attack", "Correctly Classified", "Index"]

def add_classification_idxs(model_name, ds_name, reconstruction, attack, cor_classified, index): 
    row_dict = {"Model" : [model_name],
                "Dataset" : [ds_name], 
                "Reconstruction" : [reconstruction],
                "Attack" : [attack], 
                "Correctly Classified" : [cor_classified], 
                "Index" : [index]}

    add_row_to_csv(CLASSIFICATION_IDXS_FILE, CLASSIFICATION_IDXS_COLS, row_dict)



def set_seeds(seed=42, fully_deterministic=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if fully_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

def show_image(im_data, scale=1):
    dpi = matplotlib.rcParams['figure.dpi']
    height, width = im_data.shape
    figsize = scale * width / float(dpi), scale * height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')
    ax.imshow(im_data, vmin=0, vmax=1, cmap='gray')
    plt.show()
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)


# TRAINING
def show_recon(img, *models):
    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(10 * len(models), 5))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i, model in enumerate(models):
        model.eval()
        img_ = img.unsqueeze(0).unsqueeze(0)
        recon = model.reconstruct(img_).squeeze()
        output = np.hstack([img.cpu(), np.ones([img.shape[0], 1]), recon.cpu(), np.ones([img.shape[0], 1]), np.abs((img-recon).cpu())])
        axes[i].imshow(output, vmin=0, vmax=1, cmap='gray')
        model.train()


# TRAINING
def save_img(recon, label, path, idx, is_tiled, num_tiles, file_name_suffix = ""):
    p = Path(path)
    p.mkdir(parents=True,exist_ok=True)
    print(f"recon image shape: {recon.shape}")
    file_name_suffix = "" if file_name_suffix == "" else f"_{file_name_suffix}"
    filename = f"img{label}_{idx}{file_name_suffix}.png"
    if is_tiled: 
        real_idx = idx // num_tiles
        split_num = idx % num_tiles
        filename = f"img{label}_{real_idx}_{split_num}{file_name_suffix}.png"
    matplotlib.image.imsave(p / filename, recon.cpu().numpy(), cmap = "gray")	
    checkrecon = np.asarray(Image.open(p / filename).convert("L"))
    print(f"loaded image shape: {checkrecon.shape}")


# LAYERS RECONSTRUCTION
def recon_comparison(model, ds_test, names, descriptions):
    images = []
    labels = []
    for idx in range(len(ds_test)):
        (image, label) = ds_test[idx]    
        img = image.squeeze()
        images.append(img)
        labels.append(label)

    my_orig = torch.stack(images)
    my_dataset_orig = torch.utils.data.TensorDataset(my_orig,torch.Tensor(labels))

    my_dataset_recon =[]
    for layer, name, description in zip(model, names, descriptions):
        images = []
        for idx in range(len(ds_test)):
            (image, label) = ds_test[idx]    
            img = image.to(device)
            
            for_recon = img.unsqueeze(0).to(device)
            layer.to(device).eval()
            recon = layer.reconstruct(for_recon).squeeze()
            images.append(recon.detach().cpu())

        my_recon = torch.stack(images)
        my_dataset_recon.append(torch.utils.data.TensorDataset(my_recon, torch.Tensor(labels)))
    return my_dataset_orig, my_dataset_recon



def get_rate_upper_bound(model, example_input):
    assert len(example_input.shape) == 4, "Expected (1, num_channels, x_h, x_w)"
    assert example_input.shape[0] == 1, "Please provide example with batch_size=1"
    
    z_e = model.encode(example_input)
    _, top_indices, _, _ = model.codebook(z_e)
        
    # assume worst case scenario: we have a uniform usage of all our codes
    rate_bound = top_indices[0].numel() * np.log2(model.codebook.codebook_slots)

    return rate_bound



def show_original(idx, ds_test):
    x, _ = ds_test[idx]
    image = x.squeeze()
    show_image(image)
