from Datasets.ImageNetDataset import ImageNet100
from train_hae import load_hae_from_checkpoints
from train_resnet import test_recon_model
from torchvision import transforms

#from resnet_I_18 import resnetI18
from resnet_I_50 import resnetI50
from utils import *
from torch.utils.data import random_split
from torchattacks import FGSM



import torch
torch.cuda.empty_cache()

ROOT_FOLDER = '/home/dl_class/data/ILSVRC/Data/CLS-LOC/'

hae_model = load_hae_from_checkpoints()
ds_test = ImageNet100(ROOT_FOLDER, split='val', transform=transforms.Compose([
                                                            transforms.CenterCrop(224),
                                                            transforms.ToTensor() ]))

ds_test, _ = random_split(ds_test, [len(ds_test) - int((3/4)*len(ds_test)),int((3/4)*len(ds_test))])

print(f'Loaded {len(hae_model)} layer model')


hae_model.eval()

layer_descriptions = [
    "downsample 2 in each dimension, latent space size of 16x16",
    "downsample 4 in each dimension, latent space size of 8x8",
    "downsample 8 in each dimension, latent space size of 4x4",
    "downsample 16 in each dimension, latent space size of 2x2",
    "downsample 32 in each dimension, latent space size of 1x1",
]

# Show reconstruction comparison over each layer in HQA
my_dataset_orig, my_dataset_recon = recon_comparison(hae_model, ds_test, LAYER_NAMES, layer_descriptions)
model = resnetI50(None, None, 100, RESNET50_I100_PATH)
model.load_state_dict(torch.load(RESNET50_I100_PATH))
i =0
for my_recon in my_dataset_recon:
    correct_idxs, incorrect_idxs, correct_rec_idxs, incorrect_rec_idxs, _, _ =test_recon_model(model, my_dataset_orig, my_recon, f"Layer{i}", attack = None)
    print(f"Accuracy: {len(correct_idxs)/(len(incorrect_idxs)+len(correct_idxs))}")
    print(f"Accuracy recon{i}: {len(correct_rec_idxs)/(len(incorrect_rec_idxs)+len(correct_rec_idxs))}")
    i=i+1


# Adversarial attack on ImageNet100
fgsm_attack = FGSM(model, eps = 1/255)
i = 0
for my_recon in my_dataset_recon:
    correct_idxs, incorrect_idxs, correct_rec_idxs, incorrect_rec_idxs, correct_atc, incorrect_atc =test_recon_model(model, my_dataset_orig, my_recon, f"Layer{i}", attack = fgsm_attack)
    print(f"Accuracy: {len(correct_idxs)/(len(incorrect_idxs)+len(correct_idxs))}")
    print(f"Accuracy recon{i}: {len(correct_rec_idxs)/(len(incorrect_rec_idxs)+len(correct_rec_idxs))}")
    print(f"Accuracy attack{i}: {len(correct_atc)/(len(incorrect_atc)+len(correct_atc))}")
    i=i+1


