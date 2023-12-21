[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13271031&assignment_repo_type=AssignmentRepo)

# EPFL - CS-433: project 1
### Team 
Team name : Brainwave Brigade

Team members : 
- Florian David  -  SCIPER : 375252, email: florian.david@epfl.ch
- Sophie Meuwly  -  SCIPER : 375305, email: sophie.meuwly@epfl.ch
- Iris Segard  -  SCIPER : 310860, email: iris.segard@epfl.ch

### Project description
The goal of this machine learning project is to predict brain activity during movies watching using fMRI data.
### Dataset
The data where given by the MIP Lab (https://miplab.epfl.ch/) and are confidential. The data set is composed of 14 movies and their correspondinf fMRI acquisition of 1 participant. However one movie and the corresponding masked fMRI are available on this github. This allow you to run the model.

### dataset.py
Store the data, pre processing of the movies and masks for fMRI.
### Repo architecture

#### Run.ipynb 
Training of our best model.

#### model.py
It contains all the functions related to the model, its training, and its validationEncoder and model training functions. 
For each of the 14 films, 1 out of 5 TR was extracted uniformly to constitute the testset. Then, during training, the trainset is completely shuffled to prevent overfitting. We used a batch size of 1 at each iteration, meaning that the encoder was trained on 1 TR per step for 20 epochs, and we used a learning rate of $1 \cdot 10^{-4}$. In the following, we refer to this model as the "best model".



#### visualisation.py
Functions to visualize model performances, by plotting the E_loss and cosine similyrity. Evaluate the correlation of each model. 


