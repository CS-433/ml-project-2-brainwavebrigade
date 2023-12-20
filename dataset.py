# -*- coding: utf-8 -*-

"""
This file contains all the functions used to manipulate the data (preprocessing, storing, loading).
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from IPython.display import Image,FileLink, display, IFrame
import pandas as pd
import imageio
from concurrent.futures import ThreadPoolExecutor
from skimage.transform import resize
import cv2

def load_data(vid = 'all'):
    """
    Load the fMRI data, the preprocessed video(s) and the duration(s) of the video(s). Loading the preprocessed videos can take several minutes.

        Parameters:
            vid (string): Either 'all' or the title of a specific video.

        Returns:
            masked_fMRI: If vid is 'all', masked_fMRI is a dictionary containing the masked fMRI (numpy arrays) of all films. If vid is a specific movie, masked_fMRI is the corresponding masked fMRI (numpy array). Every masked fMRI arrays have a shape of (movie_duration, 4330).
            processed_vid: If vid is 'all', processed_vid is a dictionary containing all the processed videos (numpy arrays). If vid is a specific movie, processed_vid is the corresponding processed video (numpy array). Every video array have a shape of (movie_duration, 3, 112, 112, 32).
            dur: If vid is 'all', dur is a dictionary containing the durations (int) of all films, as well as the onset and the wash duration. If vid is a specific movie, dur (int) is the duration of the corresponding movie. 
    """
    
    print('Load data:')
    masked_fMRI = load_masked_fMRI(vid)
    print('fMRI: OK')
    processed_vid = load_processed_videos(vid)
    print('Video: OK')
    dur = load_durations(vid)
    print('Duration: OK')
    print('---')
        
    return masked_fMRI, processed_vid, dur

def load_masked_fMRI(vid = 'all'):
    """
    Load the fMRI data.

        Parameters:
            vid (string): Either 'all' or the title of a specific video.

        Returns:
            datas: If vid is 'all', datas is a dictionary containing the masked fMRI (numpy arrays) of all films. If vid is a specific movie, datas is the corresponding masked fMRI (numpy array). Every masked fMRI arrays have a shape of (movie_duration, 4330).
    """

    # you can change the path of the folder containing the preprocessed fMRI
    data_path = 'masked_fMRI'

    # store the masked fMRI in the dictionary datas. The keys are the titles of the movies, the elements are the arrays
    if vid == 'all': 
        datas = {}
        # iterate through all the files of the folder
        for d in os.listdir(data_path):
            name = d[:-4]     # -4 to remove the extension '.npz' 
            file_path = os.path.join(data_path, f'{name}.npz')
            if os.path.exists(file_path):
                duration = load_durations(vid=name)
                data = np.load(file_path)['masked_fMRI']
                data = np.transpose(data[:,77:77+duration],(1,0)) # 77 to take into account the wash and the onset. Transpose to have (movie_duration, 4330).
                datas[name] = data
   
    # store the masked fMRI in the array datas
    else:
        file_path = os.path.join(data_path, f'{vid}.npz')
        if os.path.exists(file_path):
            duration = load_durations(vid=vid)
            datas = np.load(file_path)['masked_fMRI']
            datas = np.transpose(datas[:,77:77+duration],(1,0))

    return datas

def load_processed_videos(vid='all'):
    """
    Load the preprocessed video(s). Can take several minutes to run.

        Parameters:
            vid (string): Either 'all' or the title of a specific video.

        Returns:
            vids: If vid is 'all', vids is a dictionary containing all the preprocessed videos (numpy arrays). If vid is a specific movie, vids is the corresponding preprocessed video (numpy array). Every processed video arrays have a shape of (movie_duration, 3, 112, 112, 32).
    """

    # you can change the path of the folder containing the preprocessed videos
    vid_path = 'processed_videos'

    # store the preprocessed videos in the dictionary vids. The keys are the titles of the movies, the elements are the arrays
    if vid == 'all':
        vids = {}
        # iterate through all the files of the folder
        for v in os.listdir(vid_path):
            name = v[:-4]     # -4 to remove the extension '.npz' 
            file_path = os.path.join(vid_path, f'{name}.npz')
            if os.path.exists(file_path):
                data = np.load(file_path)['video_array']
                vids[name] = data
                
    # store the preprocessed video in the array vids
    else:
        file_path = os.path.join(vid_path, f'{vid}.npz')
        if os.path.exists(file_path):
            vids = np.load(file_path)['video_array']

    return vids

def load_durations(vid='all'):
    """
    Load the duration(s) of the video(s).

        Parameters:
            vid (string): Either 'all' or the title of a specific video.

        Returns:
            durations: If vid is 'all', durations is a dictionary containing the durations (int) of all films, as well as the onset and the wash duration. If vid is a specific movie, durations (int) is the duration of the corresponding movie. 
    """

    # you can change the path of the file containing the video durations
    durations_path = '/media/miplab-nas2/Data2/Michael/ml-students2023/resources/experimentals.pkl'
    
    experimentals_df = pd.read_pickle(durations_path)
    
    # the name of 1 film is not identical as in the fMRI and video files
    experimentals_df['filmduration_TR']['BetweenViewings'] = experimentals_df['filmduration_TR']['BetweenViewing']
    del experimentals_df['filmduration_TR']['BetweenViewing']

    # store the durations in the dictionary vids. The keys are the titles of the movies, wash, and onset, the elements are the integers.
    if vid == 'all':
        durations = experimentals_df

    # store the video duration in the variable durations
    else:
        durations = experimentals_df['filmduration_TR'][vid]
        
    return durations

def store_data():
    """
    Creates 2 distinct folders to store the preprocessed fMRI data and the preprocessed video(s). Storing the data can take several minutes.

        Parameters:
        
        Returns:
    """
    
    print('Store data:\n')
    store_masked_fMRI()
    print('\nAll fMRI have been stored\n')
    store_processed_videos()
    print('\nAll videos have been stored\n')
    return

def store_masked_fMRI():
    """
    Preprocesses the fMRI data and creates a folder to store them. 

        Parameters:

        Returns:
    """

    # you can change the path of the mask or the path of the fMRI data
    mask_path = '/home/chchan/Michael-Nas2/ml-students2023/resources/vis_mask.nii'
    data_path = '/media/miplab-nas2/Data2/Michael/ml-students2023/data/sub-S01'

    # extract the mask and flatten it
    img_mask = nib.load(mask_path)
    mask_3d = np.asanyarray(img_mask.dataobj)
    mask_2d = mask_3d.reshape(-1,)
    # identify the relevant voxels to keep 
    indices = np.where(mask_2d == 1)[0]

    # create the folder to store the data if it doesn't exist
    folder = 'masked_fMRI'
    os.makedirs(folder, exist_ok=True)

    # you can change the for loop(s) depending on the structure of your folder 'data_path'
    ses = ['ses-1', 'ses-2', 'ses-3', 'ses-4']
    for s in ses:   
        feats = os.listdir(os.path.join(data_path, s))[1:]
        for feat in feats:
            # extract the fMRI data
            if str(feat[17:-5]) != 'Rest':
                MNI_path = os.path.join(data_path, s, feat, 'filtered_func_data_res_MNI.nii')
                name = MNI_path[85:-36]     # movie name
                img = nib.load(MNI_path)
                data_4d = np.asanyarray(img.dataobj)

                time_points = data_4d.shape[3]
                data_2d = data_4d.reshape(-1, time_points)     # flatten the fMRI data
                masked_fMRI = data_2d[indices]     # extract the relevant voxels only

                # save the file
                file_path = os.path.join(folder, f'{name}.npz')
                np.savez(file_path, masked_fMRI=masked_fMRI)
                print(name)

                del img, data_4d, data_2d, masked_fMRI

    return

def store_processed_videos(with_OF = False):
    """
    Preprocesses the videos and creates a folder to store them. 

        Parameters:
            with_OF (Bool): If you wish to extract the optical flow of the videos as well as the RGB channels.

        Returns:
    """
    
    # you can change the path of the videos
    video_path = '/media/miplab-nas2/Data2/Michael/ml-students2023/data/Movies_cut'
    videos = os.listdir(video_path)

    # load movies durations
    durations = load_durations()

    # create the folder to store the data if it doesn't exist
    folder = 'processed_videos'
    os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist
    
    # number of workers for parallel processing
    num_workers = 20
    with ThreadPoolExecutor(max_workers=num_workers) as executor:

        for video in videos:

            # standardization of the movie names
            video_el = video.split('_')     # remove the '_'
            key_name = ''
            for el in video_el[:-1]:   # -1 to remove the extension .mp4
                key_name = str(key_name) + str(el)
                
            #some video names are not identical to the corresponding fMRI names
            if key_name == 'TearsofSteel':
                key_name = 'TearsOfSteel'
            elif key_name == 'Thesecretnumber':
                key_name = 'TheSecretNumber'

            # get the raw video
            video_reader = imageio.get_reader(os.path.join(video_path, video))
            
            # use executor for parallel processing to preprocess the video frames (resize, normalize)
            if with_OF:
                video_processed = list(executor.map(process_frame_OF, video_reader))
            else:    
                video_processed = list(executor.map(process_frame, video_reader))
            del video_reader

            # store the video in a numpy array
            video_array = np.array(video_processed)
            del video_processed

            # resample the videos (either upsampling or downsampling)
            n_frames = video_array.shape[0]
            movie_duration = durations['filmduration_TR'][key_name]
            new_n_frames = movie_duration * 32     # 32 is the aimed frame rate (32 frames per TR). It can be modified
            indices = np.linspace(0, n_frames - 1, new_n_frames).astype(int)     # remove or duplicate some frames uniformly
            video_resampled = video_array[indices]
            del video_array

            if with_OF:
                num_channels = 5     # RGB + OF
            else:
                num_channels = 3     # RGB
            # divide the video into smaller groups of 32 frames each
            video_reshaped = video_resampled.reshape(movie_duration, 32, 112, 112, num_channels)
            del video_resampled

            # transpose from shape (movie_duration, 32, 112, 112, num_channels) to shape (movie duration, num_channels, 112, 112, 32)
            video_transposed = np.transpose(video_reshaped, (0, 4, 2, 3, 1))
            del video_reshaped

            # save the file
            file_path = os.path.join(folder, f'{key_name}.npz')
            np.savez(file_path, video_array=video_transposed)
            print(key_name)     # print video name to follow the advancement of the process
            del video_transposed
    
    return

def normalize(X):
    """
    Normalizes an array.
        Parameters:
            X (array): The numpy array to normalize. Any shape accepted.
        Returns:
            normalized_X (array): The normalized array.
    """
    
    min_val = np.min(X)
    max_val = np.max(X)
    epsilon = 1e-8      # small value to avoid division by zero
    normalized_X = (X - min_val) / (max_val - min_val + epsilon)
    return normalized_X

def process_frame(frame):
    """
    Preprocesses a video frame: resize and normalize (over the RGB channels).
        Parameters:
            frame (array): The frame to resize and normalize. Shape: (frame_length, frame_width, channels).
        Returns: 
            processed_frame: The normalised frame resized to the shape (112, 112, 3).
    """
    
    target_shape = (112, 112)     # the target shape (frame_length, frame_width) can be modified.
    resized_frame = resize(frame, target_shape, anti_aliasing=True)
    processed_frame = normalize(resized_frame)
    return processed_frame

def process_frame_OF(frame):
    """
    Preprocesses a video frame: resize and normalize (over the RGB channels and the OF channels).
        Parameters:
            frame (array): The frame to resize and normalize. Shape: (frame_length, frame_width, channels).
        Returns: 
            processed_frame: The normalised frame resized to the shape (112, 112, 5).
    """
    
    target_shape = (112, 112)     # the target shape (frame_length, frame_width) can be modified.
    resized_frame = resize(frame, target_shape, anti_aliasing=True)
    processed_frame = normalize_frame_OF(resized_frame)
    gray_frame = cv2.cvtColor(np.uint8(processed_frame * 255), cv2.COLOR_RGB2GRAY)
    if 'prev_gray' not in process_frame_OF.__dict__:
        process_frame_OF.prev_gray = gray_frame
    flow = cv2.calcOpticalFlowFarneback(process_frame_OF.prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)     # the parameters can be changed
    process_frame_OF.prev_gray = gray_frame
    processed_frame = np.concatenate([processed_frame, flow], axis=-1)
    return processed_frame