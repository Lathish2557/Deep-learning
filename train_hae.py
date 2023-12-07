
from Datasets.ImageNetDataset import ImageNet100
from models.hae import HAE

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchaudio


import os


DATA_FOLDER = '/home/dl_class/data/ILSVRC/Data/CLS-LOC/'


def load_hae_from_checkpoints(
    n_layers=5,
    checkpoints_dir='hae_checkpoints'
) -> pl.LightningModule:
    """
    Load a HAE model from a directory of checkpoints for each layer
    """
    for layer in range(n_layers):
        if layer == 0:
            hae = HAE.load_from_checkpoint(
                f"{checkpoints_dir}/hae_layer{layer}.ckpt", prev_model=None)
        else:
            hae = HAE.load_from_checkpoint(
                f"{checkpoints_dir}/hae_layer{layer}.ckpt", prev_model=hae_prev)
        hae_prev = hae
    return hae


def get_dataloader(training=True) -> DataLoader:
    if training:
        dataset = ImageNet100(DATA_FOLDER, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))
    else:
        dataset = ImageNet100(DATA_FOLDER, split="val", transform=transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]))

    dataloader = DataLoader(dataset, batch_size=64, num_workers=8)
    return dataloader


def train_full_stack(training_dataloader, 
                     validation_dataloader,
                     input_feat_dim,
                     epochs, 
                     enc_hidden_sizes=[16, 16, 32, 64, 128], 
                     dec_hidden_sizes=[16, 64, 256, 512, 1024],
                     dev_run=False,
                     num_res_blocks=0):
    n_layers = len(enc_hidden_sizes)
    for layer in range(n_layers):
        if layer == 0:
            hae = HAE.init_bottom(input_feat_dim=input_feat_dim,
                                  enc_hidden_dim=enc_hidden_sizes[layer],
                                  dec_hidden_dim=dec_hidden_sizes[layer],
                                  num_res_blocks=num_res_blocks)
        else:
            hae = HAE.init_higher(hae_prev,
                                  enc_hidden_dim=enc_hidden_sizes[layer],
                                  dec_hidden_dim=dec_hidden_sizes[layer])

        print(f"Training layer {layer}:")

        if layer == 0:
            trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=dev_run)
        else:
            trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=dev_run,
                                 strategy='ddp_find_unused_parameters_true')

        trainer.fit(hae, train_dataloaders=training_dataloader,
                    val_dataloaders=validation_dataloader)
        trainer.save_checkpoint(f"hae_checkpoints/hae_layer{layer}.ckpt")
        hae_prev = hae
    return hae


def train(epochs=20, dev_run=False, checkpoint_dir='hae_checkpoints') -> pl.LightningModule:
    """
    Train hae model on ImageNet100
    """
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    torch.set_float32_matmul_precision('medium')

    training_dataloader = get_dataloader()
    validation_dataloader = get_dataloader(training=False)

    hae = train_full_stack(training_dataloader, validation_dataloader, input_feat_dim=3, epochs=epochs, dev_run=dev_run)
    return hae

if __name__ == '__main__':
    train()

