import logging, os, sys, yaml

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import pandas as pd
import numpy as np
from tqdm import tqdm

from Models import *
from Datasets import STD_Dataset
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Function to load YAML config file into a Python dict
def load_parameters(yaml_path):
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config

# Function to create PyTorch dataset objects from
# list of datasets defined in YAML file (and loaded into a dict via load_parameters function)
def load_std_datasets(datasets, apply_vad):
    return {
        ds_name:STD_Dataset(
            root_dir = ds_attrs['root_dir'],
            labels_csv = ds_attrs['labels_csv'],
            feats_scp = ds_attrs['feats_scp'],
            apply_vad = apply_vad,
            max_height = ds_attrs['max_height'],
            max_width = ds_attrs['max_width'],
        ) for (ds_name, ds_attrs) in datasets.items()
    }

# Function to create PyTorch DataLoaders from PyTorch datasets
# created via load_std_datasets function
def create_data_loaders(loaded_datasets, config):
    return {
        ds_name:DataLoader(
            dataset = dataset,
            batch_size = config['datasets'][ds_name]['batch_size'],
            shuffle = True if ds_name == 'train' else False,
            num_workers = config['dl_num_workers']
        ) for (ds_name, dataset) in loaded_datasets.items()
    }

# Function to load saved models to continue training from or for evaluation on test data
# Expected input is a config dict with the model name (ConvNet, VGG, ResNet34) to paths(s)
# to saved models.
def load_saved_model(config):

    model, optimizer, criterion, scheduler = instantiate_model(config)

    logging.info(" Loading model from '%s'" % (config['model_path']))
    checkpoint = torch.load(config['model_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if(config['mode'] == 'train'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if config['mode'] == 'eval' and config['use_gpu'] and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model, optimizer, criterion, scheduler

# Admin function to create output directory if necessary, set up log files, make a copy
# of the config file, and create output CSV file of training predictions (if mode is training)
def setup_exp(config):
    output_dir = config['artifacts']['dir']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if(config['mode'] == 'train'):
        make_results_csv(os.path.join(output_dir, 'train_results.csv'))

    logging.basicConfig(
        filename = os.path.join(output_dir, config['artifacts']['log']),
        level = logging.DEBUG,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    return output_dir

def instantiate_model(config):
    # Instantiate model based on string given in config file (ConvNet, VGG11, ResNet34)
    constructor = globals()[config['model_name']]
    max_height = config['datasets'][config['mode']]['max_height']
    max_width = config['datasets'][config['mode']]['max_width']
    model = constructor(max_height, max_width)

    logging.info(" Instantiating model '%s'" % (config['model_name']))

    if config['use_gpu']:
        model.cuda()

    if(config['mode'] == 'train'):
        model.train()

        if(config['optimizer'] == 'adam'):
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) 

        if(config['criterion'] == 'BCELoss'):
            criterion = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1, last_epoch=-1)
        return model, optimizer, criterion, scheduler

    if(config['mode'] == 'eval'):
        model.eval()
        
        return model, None, None, None

# Create CSV file with appropriate header for training/evaluation process to then append to.
def make_results_csv(csv_path, headers = 'train'):
    if (headers == 'train'):
        csv_cols = ['epoch', 'query','reference','label','pred']
    elif (headers == 'eval'):
        csv_cols = ['query','reference','label','pred']

    t_df = pd.DataFrame(columns=csv_cols)
    t_df.to_csv(csv_path, index = False)
    return csv_path

# Append to a pre-existing output CSV file
def append_results_csv(csv_path, results_dict):
    df = pd.DataFrame(results_dict)
    df.to_csv(csv_path, mode = 'a', header = False, index = False)

# Save model at checkpoints along with optimizer state etc so you can resume training
# later
def save_model(epoch, model, optimizer, loss, output_dir, name = 'model.pt'):
    cps_path = os.path.join(output_dir, 'checkpoints')
    cp_name  = "model-e%s.pt" % (str(epoch).zfill(3))

    if not os.path.exists(cps_path):
        os.makedirs(cps_path)

    logging.info(" Saving model to '%s/%s'" % (cps_path, cp_name))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, os.path.join(cps_path, cp_name)
    )
 
# Helper function for running models (both training and evaluation)
def run_model(model, mode, ds_loader, use_gpu, csv_path, keep_loss, criterion, optimizer, epoch):
    if keep_loss is True:
        total_loss = 0
    else:
        # Set to None for referencing in return
        loss = None

    # Iterate over mini batches suppied by the dataloader
    for batch_index, batch_data in enumerate(tqdm(ds_loader)):
        dists  = batch_data['dists']
        labels = batch_data['labels']

        # Move data to GPU if desired
        if (use_gpu):
            dists, labels = dists.cuda(), labels.cuda()

        # If training, set model to training mode and clear any accumulated gradients
        if(mode == 'train'):
            model.train()
            optimizer.zero_grad()
        
        # Otherwise, set model to evaluation mode 
        elif(mode == 'eval'):
            model.eval()

        # Make predictions
        outputs = model(dists)

        # Add to epoch loss if keeping track of loss (e.g. for Dev data)
        if keep_loss is True:
            loss        = criterion(outputs, labels)
            total_loss += loss.cpu().data

            # If in training mode, do backprop and optimisation step
            if(mode == 'train'):
                loss.backward()
                optimizer.step()

        # Prepare to append to output CSV file, e.g.
        ## | epoch | query | reference | label | pred |
        ## |   1   | word1 | sentence1 |   1   | 0.99 |
        batch_output = {}

        if(epoch is not None):
            # Repeat epoch for number of rows needed data frame
            # e.g. epoch = 1 = [1, 1, ... 1, 1]
            batch_output['epoch'] = [epoch] * len(batch_data['query'])

        batch_output['query'] = batch_data['query']
        batch_output['reference'] = batch_data['reference']
        batch_output['label'] = batch_data['labels'].reshape(-1).numpy().astype(int)
        batch_output['pred'] = outputs.cpu().detach().reshape(-1).numpy().round(10)

        # Append to CSV
        append_results_csv(csv_path, batch_output)

    if keep_loss is True:
        mean_loss = total_loss / len(ds_loader)
    else:
        # Set to None for referencing in return
        mean_loss = None

    return model, optimizer, criterion, loss, mean_loss
