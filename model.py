# -*- coding: utf-8 -*-

"""
This script contains all the functions related to the model, its training, and its validation
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show
from sklearn.decomposition import FastICA, PCA
from nilearn import datasets, plotting
from scipy.stats import zscore
from scipy.io import loadmat
from IPython.display import Image,FileLink, display, IFrame
import pandas as pd
import torch
import torch.nn as nn
import imageio
import time
from concurrent.futures import ThreadPoolExecutor
from skimage.transform import resize
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from dataset import * 
from visualisation import *


class Encoder(nn.Module):
    def __init__(self):
        """
        Encoder architecture. Inspired by the work of Kupershmidt et al. (2022).
        """
        
        super(Encoder, self).__init__()

        # 3D convolutional layer 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 1), padding=(1, 1, 0), padding_mode='zeros'),
            nn.MaxPool3d(kernel_size=(2, 2, 1))
        )

        # 3D convolutional layer 2
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 256, kernel_size=(1, 1, 5), padding=(0, 0, 2)),
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            nn.AvgPool3d((1, 1, 2))
        )

        # 16x1x1 temporal combinations
        num_combinations = 16
        self.temporal_combinations = nn.ModuleList()
        for _ in range(num_combinations):
            combination = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(128)
            )
            self.temporal_combinations.append(combination)

        # 2D convolutional layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # flatten + dropout
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5)    # hyperparameter ?
        )

        # fully connected layer
        self.fc = nn.Linear(12544, 4330)

    def forward(self, x):
        """
        Forward pass of the Encoder.
        
        Parameters:
            x (tensor): Preprocessed video (input). Shape: (movie_duration, 3, 112, 112, 32).
        Returns: 
            x (tensor): Predicted fMRI signal (prediction). Shape: (movie_duration, 4330)
        """
        
        # 3D convolutional layer 1
        x = self.conv1(x)

        # 3D convolutional layer 2
        x = self.conv2(x)

        # 16x1x1 temporal combinations
        tensor = []
        for i in range(len(self.temporal_combinations)):
            t = self.temporal_combinations[i](x[:, :, :, :, i])
            tensor.append(t)
        tensor = torch.cat(tensor, dim=1)

        # 2D convolutional layer
        x = self.conv3(tensor)

        # flatten + dropout
        x = self.flatten(x)

        # fully connected layer
        x = self.fc(x)

        return x

class E_Loss(nn.Module):
    def __init__(self, alpha=0.5):
        super(E_Loss, self).__init__()
        self.alpha = alpha

    def forward(self, prediction, label):
        """
        Calculation of the Encoder loss. Combination of MSE and cosine similarity (or cosine distance). The aim is to minimise the MSE while maximising the cosine similarity between the prediction and ground truth (or label).

        Parameters:
            prediction (tensor): Output of the Encoder when given a film TR. Shape: (1, 4330).
            label (tensor): Ground truth (real fMRI data). Shape: (1, 4330)
        Returns:
            loss (int): Combined loss (MSE and cosine similarity) between the prediction and the ground truth.
            cos_sim (int): Cosine similarity between the prediction and the ground truth.
        """
        
        # mean squared error. Divide by the target shape to rescale and give the same importance to both mse_loss and cos_sim
        mse_loss = F.mse_loss(prediction, label)/label.shape[1]
        # cosine similarity
        cos_sim = F.cosine_similarity(prediction, label, dim=1)
        # combined loss
        loss = mse_loss + self.alpha * (1 - cos_sim).mean()

        return loss, cos_sim

def split_data(video_titles, shuffle):
    """
    Split the dataset into a trainset and testset. Here, the proportions are 0.8:0.2 by default.

        Parameters:
            video_titles (list): List of the movies to use. This list contains the titles (string) of the movies.
            shuffle (Bool): If True, the TR are shuffled before splitting.
        Returns:
            train_input (array): Inputs of the trainset, no distinction are made between the movies. Shape: (trainset_size, 3, 112, 112, 32).
            train_label (array): Labels of the trainset, no distinction are made between the ground truths. Shape: (trainset_size, 4330).
            test_inputs (dict): Inputs of the testset, each movies is stored individually. The keys of the dictionary are the movies titles, and the elements are the input arrays each of shape (movie_testset_size, 3, 112, 112, 32).
            test_labels (dict): Labels of the testset, each ground truth is stored individually. The keys of the dictionary are the movies titles, and the elements are the labels arrays each of shape (movie_testset_size, 4330).
    """
    
    tic = time.time()
    
    # load the data
    fMRIs, videos, durations = load_data()

    print('Spliting:')
    train_input = None
    test_inputs, test_labels = {}, {}
    for vid in video_titles: 
        print(vid)    # to follow the advancement of the splitting
        # total number of samples (total number of TR, or movie_duration)
        total_samples = videos[vid].shape[0]

        if shuffle:
            indices = np.arange(total_samples)
            np.random.seed(1)     # for reproducibility
            np.random.shuffle(indices)
            sep = int(np.ceil(0.8*total_samples))     # 0.8: 80% of the samples constitute the trainset. you can modify this value
            train_indices = indices[:sep]
            test_indices = indices[sep:]
        
        # best model
        else:
            train_indices = [i for i in range(total_samples) if i % 5 != 0]     # 5: 80% of the samples constitute the trainset. you can modify this value
            test_indices = [i for i in range(total_samples) if i % 5 == 0]

        # the trainset is an array
        if train_input is None:
            train_input = videos[vid][train_indices]
            train_label = fMRIs[vid][train_indices]
        else:
            train_input = np.concatenate((train_input, videos[vid][train_indices]),axis=0)
            train_label = np.concatenate((train_label, fMRIs[vid][train_indices]),axis=0)

        # the testset is a dictionary
        test_inputs[vid] = videos[vid][test_indices]
        test_labels[vid] = fMRIs[vid][test_indices]
    
    del fMRIs, videos, durations
    
    print("Split time (minutes):", (time.time()-tic)/60)
    print('---')
    return train_input, train_label, test_inputs, test_labels

def train_model(input, label, num_epochs, lr, batch_size, device, save_model_as):
    """
    Train the model with Adam algorithm. Display the training E_Loss and the training cosine similarity.

        Parameters:
            input (array): Inputs of the trainset. Shape: (trainset_size, 3, 112, 112, 32).
            label (array): Labels of the trainset. Shape: (trainset_size, 4330).
            num_epochs (int): Number of epochs to train.
            lr (float): Learning rate of Adam algorithm.
            batch_size (int): Size of a minibatch for training.
            device (string): either 'cpu' or 'cuda'.
            save_model_as (string): Name of the model to store it in a file of the same name.
        Returns:
            model (torch.nn.Module): Trained model with updated parameters
    """
    
    tic = time.time()
    input = torch.from_numpy(input)
    label = torch.from_numpy(label)
    print('Training on input of shape', input.shape) 
    
    train_set = TensorDataset(input, label)
    del input, label
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,     # shuffle the TR to prevent overfitting
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        num_workers=50,
    )
    
    # set the model (Encoder), the optimizer (Adam) and the criterion (E_Loss)
    model = Encoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = E_Loss(alpha=0.5)#.to(device)

    # store the train loss and cos_sim
    train_loss_history = []
    train_cos_sim_history = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_cos_sim = train_epoch(
            model, device, train_loader, optimizer, epoch, criterion
        )
        train_loss_history.extend(train_loss)
        train_cos_sim_history.extend(train_cos_sim)

    # plot train loss and cos_sim
    n_train = len(train_loss_history)
    t_train = num_epochs * np.arange(n_train) / n_train

    plt.figure()
    plt.plot(t_train, train_loss_history, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Evolution of E_loss across epoch, average over 10 minibatches") 

    plt.figure()
    plt.plot(t_train, train_cos_sim_history, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Cos sim")
    plt.title("Evolution of cosine similarity across epoch, average over 10 minibatches") 

    # save the model parameters for later use
    torch.save(model.state_dict(), save_model_as)
    print("Train time (minutes):", (time.time()-tic)/60)
    print('---')
    return model

def train_epoch(model, device, train_loader, optimizer, epoch, criterion):
    """
    Train the model with Adam algorithm. Display the training E_Loss and the training cosine similarity.

        Parameters:
            model (torch.nn.Module): Model to train.
            device (string): either 'cpu' or 'cuda'.
            train_loader (torch.utils.data.DataLoader): Trainset.
            optimizer (torch.optim.Optimizer): Optimizer (Adam here).
            epoch (int): Index of the current epoch.
            criterion (torch.nn.modules.loss._Loss): Objective for training (E_Loss here).
        Returns:
            loss_history (array): Contains the average E_Losses over 10 minibatches. Shape: (N,)
            cos_sim_history (array): Contains the average cosine similarities over 10 minibatches. Shape: (N,)
    """

    # set the model in the "training phase"
    model.train()  

    # store the E_losses and cosine similarities
    c, average_loss, average_cos_sim = 0, [], []
    loss_history, cos_sim_history = [], []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data.float())
        loss, cos_sim = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store temporarly 10 losses and cos_sims
        if c < 10:
            average_loss.append(loss.item())
            average_cos_sim.append(cos_sim.item())
            c += 1
            
        # average the 10 losses and cos_sims that have been stored
        else:
            average_loss.append(loss.item())
            loss_history.append(np.mean(average_loss))
            average_cos_sim.append(cos_sim.item())
            cos_sim_history.append(np.mean(average_cos_sim))
            c, average_loss, average_cos_sim = 0, [], []

        # follow the advancement of the process by printing the first minibatch E_loss and cos_sim of each epoch
        if batch_idx == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx} batch_loss={loss.item()/len(data):0.2e}, batch_cos_sim=={cos_sim.item()/len(data):0.2e}"
            )

    return loss_history, cos_sim_history

def test_model(inputs, labels, model, plot_individual_results = False):
    """
    Test the model performances on the testset.

        Parameters:
            inputs (dict): Inputs of the testset, each movie is stored individually. The keys of the dictionary are the movies titles, and the elements are the input arrays each of shape (movie_testset_size, 3, 112, 112, 32).
            labels (dict): Labels of the testset, each ground truth is stored individually. The keys of the dictionary are the movies titles, and the elements are the labels arrays each of shape (movie_testset_size, 4330).
            model (torch.nn.Module): Trained model with updated parameters.
            num_epochs (int): Number of epochs to train.
            plot_individual_results (Bool): If True, than it will display the model performance on every movie and not only the overall performance.
        Returns:
            predictions (dict): Predictions of the testset, each prediction is stored individually. The keys of the dictionary are the movies titles, and the elements are the model's output arrays each of shape (movie_testset_size, 4330).
            E_Losses (array): Contains the E_Losses calculated between the predictions and the labels. Shape (N,).
            cos_sims (array): Contains the cosine similarities calculated between the predictions and the labels. Shape (N,).
    """

    print('Start testing:')
    tic = time.time()
    device = 'cpu'
    criterion = E_Loss(alpha=0.5)
    model = model.to(device)

    # store the predictions, E_Losses, and cosine similarities
    predictions = {}
    E_Losses, cos_sims = [], []
    # iterate through all videos, one at a time
    for vid in inputs.keys():
        input = inputs[vid]
        label = labels[vid]

        input = torch.from_numpy(input).to(device)
        label = torch.from_numpy(label).to(device)

        test_set = TensorDataset(input, label)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=1,     # 1 TR at a time
            shuffle=False,    # to keep the TR order for later use
            pin_memory=torch.cuda.is_available(),
            num_workers=20,
        )

        # set the model in the "testing phase"
        model.eval()
        
        # make a prediction of the whole test input
        with torch.no_grad():
            predictions[vid] = model(input.float())
        del input, label

        # compute the E_Loss and cosine similarity for each individual TR
        test_loss_history, test_cos_sim_history = [], []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
    
            with torch.no_grad():
                output = model(data.float())
                loss, cos_sim = criterion(output, target)
                test_loss_history.append(loss.item())
                test_cos_sim_history.append(cos_sim.item())

        # plot the E_Loss and cosine similarity for the current movie
        if plot_individual_results:
            plot_Eloss_cossim(test_loss_history, test_cos_sim_history, vid)
        
        E_Losses = np.concatenate((E_Losses, test_loss_history))
        cos_sims = np.concatenate((cos_sims, test_cos_sim_history))

    # plot the overall E_Loss and cosine similarity (across all movies)
    plot_Eloss_cossim(E_Losses, cos_sims, None)

    # plot the overall correlations (across all movies)
    #plot_correlations(predictions, test_labels, False)
    
    print("Test time (minutes):", (time.time()-tic)/60)
    print('---')
    return predictions, E_Losses, cos_sims