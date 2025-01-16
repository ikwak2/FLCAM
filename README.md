# F-LCAM : Frequency Light Convolutional Attention Module for Sound Event Detection

This repository contains the source for a manuscript "F-LCAM : Frequency Light Convolutional Attention Module for Sound Event Detection."

![SED_Overview](https://github.com/user-attachments/assets/d064d6d5-4b37-412d-ae18-5198385a3e98)

A PDF will be available after publication.


## Data Preparation
[DCASE 2022 Task4 Description](https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments#download) 페이지를 참고하여 데이터 셋을 다운로드 받을 수 있습니다. 
다운로드 후, config.yaml 파일에서 데이터셋 경로를 지정해야합니다.

## File Description
* main.py : train/validation/test 메인 스크립트
* configs
    * config.yaml : 모델, 데이터셋, training 학습 설정 파일
* utils
    * dataset.py : Strong/Weak/Unlabeled data 데이터셋 로더
    * data_aug.py : Data Augmentation methods
    * evaluation_measures.py : 평가 지표 계산 (ex. PSDS1/PSDS2, F1 score, etc)
    * settings_CRNN.py : 데이터 로딩, 로깅 및 저장
    * utils.py : 유틸리티 함수(encoder, learning rate scheduler, etc)
    * flcam 및 exp1~5는 어떻게 적을지...

## Installation
Python version : 3.7.10
* pytorch==1.8.0
* pytorch-lightning==1.2.4
* pytorchaudio==0.8.0
* scipy==1.4.1
* pandas==1.1.3
* numpy==1.19.2



## Training
`config.yaml`에서 저장 경로를 지정하고 training을 하면 `exp\` 디렉토리에 저장됩니다.
```
python main.py
```




