# -*- coding: utf-8 -*-

"""
This script contains all the functions to plot results, video frames, and fMRI
"""


import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from IPython.display import Image,FileLink, display, IFrame
import pandas as pd
import imageio
from concurrent.futures import ThreadPoolExecutor
from skimage.transform import resize
import seaborn as sns


def plot_Eloss_cossim(E_loss, cos_sim, vid):
    """
    Plot the E_Loss and the cosine similarity between the prediction and the label for a specific video or for all videos at once without any distinction (overall). The plots are histograms to observe the distribution of the two metrics computed for each individual TR.

        Parameters:
            E_Loss (array): Contains the E_Loss calculated between the prediction(s) and the label(s) for 1 movie or all of them. Shape (N,).
            cos_sim (array): Contains the cosine similarity calculated between the prediction(s) and the label(s) for 1 movie or all of them. Shape (N,).
            vid (string or None): It must correspond to the nature of E_Loss and cos_sim (either the title of a specific video or None if you want to plot all of them at once
        Returns:
    """
    
    E_loss = np.asarray(E_loss)
    cos_sim = np.asarray(cos_sim)

    plt.figure(figsize=(11, 5))

    # plot E_Loss as a histogram
    plt.subplot(1, 2, 1)
    # define the bins, you can be more precise here
    bins = [0, 0.5, 1, 1.5, 2, 2.5, max(np.max(E_loss),3)]     # "max(np.max(E_loss),3)" in case the maximal value is < 2.5
    lab = ['[0, 0.5]', ']0.5, 1]', ']1, 1.5]', ']1.5, 2]', ']2, 2.5]', '> 2.5']
    # define the histogram and calculate the frequence of each bin 
    hist, bins = np.histogram(E_loss, bins=bins)
    freq = [(count / E_loss.shape[0]) * 100 for count in hist]
    colormap = plt.cm.get_cmap('RdYlGn_r', len(lab))
    plt.bar(lab, freq,color=colormap(np.arange(len(lab))))
    # display the number of samples (number of TR), the mean and the median of the metric
    plt.axhline(0, color='k', linestyle=None,label=f'N={E_loss.shape[0]:.0f}', linewidth=0)
    plt.axhline(0, color='k', linestyle=None,label=f'Mean={np.mean(E_loss):.3f}', linewidth=0)
    plt.axhline(0, color='k', linestyle=None,label=f'Median={np.median(E_loss):.3f}', linewidth=0)
    plt.rcParams['legend.handlelength'] = 0
    plt.legend()
    plt.xlabel('E_Loss')
    plt.ylabel('Proportion (%)')
    if vid == None:
        plt.title('E_Loss between predicted TR and ground truth\nOverall')
    else:
        plt.title(f'E_Loss between predicted TR and ground truth\nVideo: {vid}')

    # plot cosine similarity as a histogram
    plt.subplot(1, 2, 2)
    # define the bins, you can be more precise here
    bins = [-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
    lab = ['< 0', ']0, 0.2]', ']0.2, 0.4]', ']0.4, 0.6]', ']0.6, 0.8]', ']0.8, 1]']
    # define the histogram and calculate the frequence of each bin 
    hist, bins = np.histogram(cos_sim, bins=bins)
    freq = [(count / np.asarray(cos_sim).shape[0]) * 100 for count in hist]
    colormap = plt.cm.get_cmap('RdYlGn', len(lab))
    plt.bar(lab, freq,color=colormap(np.arange(len(lab))))
    # display the number of samples (number of TR), the mean and the median of the metric
    plt.axhline(0, color='k', linestyle=None,label=f'N={cos_sim.shape[0]:.0f}', linewidth=0)
    plt.axhline(0, color='k', linestyle=None,label=f'Mean={np.mean(cos_sim):.3f}', linewidth=0)
    plt.axhline(0, color='k', linestyle=None,label=f'Median={np.median(cos_sim):.3f}', linewidth=0)
    plt.rcParams['legend.handlelength'] = 0
    plt.legend()
    plt.xlabel('Cosine similarity')
    plt.ylabel('Proportion (%)')
    if vid == None:
        plt.title('Cosine similarity between predicted TR and ground truth\nOverall')
    else:
        plt.title(f'Cosine similarity between predicted TR and ground truth\nVideo: {vid}')
    
    plt.tight_layout()
    plt.show()

def plot_correlations(predictions, test_labels, permutation_testing):
    """
    Plot the distribution of correlations between the predictions and the labels in 2 different manners: with respect to the TR (all the voxels at a certain time point) or with respect to the voxel (the time course of a specific voxel). 

        Parameters:
            predictions (dict): Predictions of the testset, each prediction is stored individually. The keys of the dictionary are the movies titles, and the elements are the model's output arrays each of shape (movie_testset_size, 4330).
            test_labels (dict): Labels of the testset, each ground truth is stored individually. The keys of the dictionary are the movies titles, and the elements are the labels arrays each of shape (movie_testset_size, 4330).
            permutation_testing (Bool): If True, a permutation test will be performed to assess the significance of the correlations.
        Returns:
    """

    # concatenate all the predictions and all the labels
    tot_pred = None
    for k in predictions.keys():
        if tot_pred is None:
            tot_pred = predictions[k].numpy()
            tot_lab = test_labels[k]
        else:
            tot_pred = np.concatenate((tot_pred, predictions[k].numpy()))
            tot_lab = np.concatenate((tot_lab, test_labels[k]))

    # calculate correlations over voxels for each TR
    correlations_over_voxel = np.zeros(tot_lab.shape[0])
    for i in range(tot_lab.shape[0]):
        correlations_over_voxel[i] = np.corrcoef(tot_lab[i, :], tot_pred[i, :])[0, 1]

    # calculate correlations over time for each voxel
    correlations_over_time = np.zeros(tot_lab.shape[1])
    for i in range(tot_lab.shape[1]):
        correlations_over_time[i] = np.corrcoef(tot_lab[:, i], tot_pred[:, i])[0, 1]
    
    # plotting the correaltions distributions
    plt.figure(figsize=(13, 5))
    plt.suptitle('Correlations distributions')

    # with respect to a specific TR
    plt.subplot(1,2,1)
    # display the number of samples (number of TR)
    plt.axhline(0, color='k', linestyle=None,label=f'N={len(correlations_over_voxel):.0f}', linewidth=0)
    # kernel distribution
    sns.kdeplot(correlations_over_voxel, fill=True, color='g' ,label='Correlations')
    plt.xlabel('Correlation Over Voxel')
    plt.ylabel('Density')
    # plot also a normal distribution as reference
    mean = np.mean(correlations_over_voxel)
    std = np.std(correlations_over_voxel)
    x = np.linspace(mean - 3*std, mean + 3*std, 1000)
    plt.plot(x, stats.norm.pdf(x, mean, std), 'k', label=f'Gaussian\nN({mean:.3f},{std:.3f}²)')
    plt.axvline(mean, color='r', linestyle='--', linewidth=1)
    plt.rcParams['legend.handlelength'] = 1
    plt.legend(loc='upper right')

    # with respect to a specific voxel
    plt.subplot(1,2,2)
    # display the number of samples (number of voxels)
    plt.axhline(0, color='k', linestyle=None,label=f'N={len(correlations_over_time):.0f}', linewidth=0)
    # kernel distribution
    sns.kdeplot(correlations_over_time, fill=True, color='b' ,label='Correlations')
    plt.xlabel('Correlation Over Time')
    plt.ylabel('Density')
    # plot also a normal distribution as reference
    mean = np.mean(correlations_over_time)
    std = np.std(correlations_over_time)
    x = np.linspace(mean - 3*std, mean + 3*std, 1000)
    plt.plot(x, stats.norm.pdf(x, mean, std), 'k', label=f'Gaussian\nN({mean:.3f},{std:.3f}²)')
    plt.axvline(mean, color='r', linestyle='--', linewidth=1)
    plt.rcParams['legend.handlelength'] = 1
    plt.legend(loc='upper right')
    
    plt.show()

    # to assess the significance of each individual correlation
    if permutation_testing:
        p_values_TR = permutation_test(correlations_over_voxel, tot_pred, tot_lab)
        p_values_voxel = permutation_test(correlations_over_time, tot_pred.T, tot_lab.T)
        
        plt.figure(figsize=(13,2))
        sns.kdeplot(p_values_TR, fill=True, color='g' ,label=f'Correlations over voxels (N={p_values_TR.shape[0]})')
        sns.kdeplot(p_values_voxel, fill=True, color='b' ,label=f'Correlations over TR (N={p_values_voxel.shape[0]})')
        plt.xlim([0,1])
        plt.xlabel('p-value')
        plt.xticks([i*0.1 for i in range(11)])
        plt.legend()
        plt.title('p-values distribution (permutation test)')
        plt.show()

def permutation_test(correlations, tot_pred, tot_lab):
    """
    Plot the distribution of p-values computed for each individual correlation during a permutation test.
    The null hypothesis is based on the assumption that there is no significant correlation between the predicted and labelled values beyond what could occur by chance. The permutation test randomly shuffles the labels while maintaining the order of the predictions. Through multiple iterations (1000 permutations here), the test recalculates correlations between the shuffled labels and the predicted values, and generates a distribution of correlation coefficients under the null hypothesis. This distribution is then used to compare the observed correlations, and for each observation we compute a p-value to determine the likelihood of obtaining such a correlation by chance alone.

        Parameters:
            correlations (array): List of all the correlations between the predictions and the labels (w.r.t voxels or TR). Shape: (N,).
            tot_pred (array): All the predictions. Shape: (N,).
            tot_lab (array): All the labels. Shape: (N,).
        Returns:
            p_values (array): p-value of each individual correlation. Shape: (N,).
    """

    # number of permutations. this value can be modified
    num_permutations = 1000  
    
    # store the permutation test statistics
    perm_test_stats = []
    for s in range(num_permutations):
        # shuffle the labels
        #np.random.seed(s)
        np.random.shuffle(tot_lab)
    
        # recompute correlations for each pair after shuffling
        perm_correlations = []
        for i in range(tot_pred.shape[0]):
            corr = np.corrcoef(tot_pred[i], tot_lab[i])[0, 1]
            perm_correlations.append(corr)
        perm_test_stats.append(perm_correlations)

    observed_stats = np.array(correlations)
    perm_test_stats = np.array(perm_test_stats)
    observed_stats = observed_stats[:, None]
    perm_test_stats = perm_test_stats.T       # transpose to match shapes for comparison
    
    # calculate the p-value for each observed correlation
    p_values = np.mean(perm_test_stats >= observed_stats, axis=1)
    #count = (p_values < 0.05).sum()      # find how many of the correlations have a p-value below a specific threshold
    return p_values

def plot_frame(vid='BetweenViewings', TR=3, frame=3):
    """
    Plot a specific movie frame after preprocessing.

    Parameters:
        vid (string): The title of a specific film.
        TR (int): The movie timepoint. Must be < movie_duration.
        frame (int): The frame of a specific TR. Must be < 32.
    Returns:
    """
    
    _, video, _ = load_data(vid)
    img = np.transpose(video[TR, :, :, :, frame], (1, 2, 0)) # to have (length, width, channels)
    plt.imshow(img)
    plt.title(f'Preprocessed {vid}, frame {frame}, TR {TR}')
    plt.axis('off')
    plt.show()

def plot_brain():
    """
    Plot 3 slices of the full brain, mask, and masked brain for a specific movie. Predifined function. 
        Parameters:
        Returns:
    """
    
    # you can choose other brain slices
    TR = 120
    x, y, z = 69, 24, 28

    # you can change the path
    mask_path = '/home/chchan/Michael-Nas2/ml-students2023/resources/vis_mask.nii'
    img_mask = nib.load(mask_path)
    mask_3d = np.asanyarray(img_mask.dataobj)
    mask_2d = mask_3d.reshape(-1,)
    indices = np.where(mask_2d == 1)[0]    # get the relevant voxels

    # you can change the path
    data_path = '/media/miplab-nas2/Data2/Michael/ml-students2023/data/sub-S01/ses-1/pp_sub-S01_ses-1_BigBuckBunny.feat/filtered_func_data_res_MNI.nii'
    img = nib.load(data_path)
    data_4d = np.asanyarray(img.dataobj)
    time_points = data_4d.shape[3]
    data_2d = data_4d.reshape(-1, time_points)     # flatten
    masked_fMRI = data_2d[indices]      # mask the fMRI to only extract the relevant voxels

    # from the masked_fMRI flattened, come back to a 3D volume
    zer1 = np.zeros(mask_3d.shape)
    print(zer1.shape)
    print(masked_fMRI.shape)
    zer1[mask_3d ==1]=masked_fMRI[:,TR]

    plt.figure(figsize=(10,10))
    # plot 3 slices of the full brain
    plt.subplot(3,3,1)
    plt.imshow(data_4d[x,:,:,TR], cmap='gray')
    plt.subplot(3,3,2)
    plt.imshow(data_4d[:,y,:,TR], cmap='gray')
    plt.subplot(3,3,3)
    plt.imshow(data_4d[:,:,z,TR], cmap='gray')
    # plot 3 slices of the mask alone
    plt.subplot(3,3,4)
    plt.imshow(mask_3d[x,:,:], cmap='gray')
    plt.subplot(3,3,5)
    plt.imshow(mask_3d[:,y,:], cmap='gray')
    plt.subplot(3,3,6)
    plt.imshow(mask_3d[:,:,z], cmap='gray')
    # plot 3 slices of the masked brain
    plt.subplot(3,3,7)
    plt.imshow(zer1[x,:,:], cmap='gray')
    plt.subplot(3,3,8)
    plt.imshow(zer1[:,y,:], cmap='gray')
    plt.subplot(3,3,9)
    plt.imshow(zer1[:,:,z], cmap='gray')
    
    plt.show()