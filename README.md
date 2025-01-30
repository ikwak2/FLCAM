# Enhancing Sound Event Detection System with Frequency-Aware Enhancements and Semi-Supervised Learning

This repository contains the source for a manuscript "Enhancing Sound Event Detection System with Frequency-Aware Enhancements and Semi-Supervised Learning."

![SED_Overview](https://github.com/user-attachments/assets/d064d6d5-4b37-412d-ae18-5198385a3e98)

A PDF will be available after publication.


## Data Preparation
You can download the dataset by referring to the [DCASE 2022 Task 4 Description](https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments#download) page.  
After downloading, you need to specify the dataset path in the `config.yaml` file.

## File Description
* `main.py` : Main script for train/validation/test  
* `configs`  
    * `config.yaml` : Configuration file for model, dataset, and training settings  
* `utils`  
    * `dataset.py` : Dataset loader for Strong, Weak, and Unlabeled data  
    * `data_aug.py` : Data augmentation methods  
    * `evaluation_measures.py` : Evaluation metric calculations (e.g., PSDS1/PSDS2, F1 score, etc.)  
    * `settings_CRNN.py` : Data loading, logging, and saving  
    * `utils.py` : Utility functions (e.g., encoder, learning rate scheduler, etc.)  
* `flcam` and `exp1~5` : [Provide a brief description of these directories/files]  

## Installation
Python version : 3.7.10
* pytorch==1.8.0
* pytorch-lightning==1.2.4
* pytorchaudio==0.8.0
* scipy==1.4.1
* pandas==1.1.3
* numpy==1.19.2



## Training
Specify the save path in `config.yaml`, and when training is executed, the results will be saved in the `exp\` directory.  

Run the following command to start training:  
```
python main.py
```

---

The manuscript is licensed under the
[Creative Commons Attribution 3.0 Unported License](http://creativecommons.org/licenses/by/3.0/).

[![CC BY](http://i.creativecommons.org/l/by/3.0/88x31.png)](http://creativecommons.org/licenses/by/3.0/)

The software is licensed under the [GNU license](License.md).
