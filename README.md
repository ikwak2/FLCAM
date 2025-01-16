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
    * settings.py : 모델 설정, 데이터 로딩, 로깅 및 저장
    * utils.py : 유틸리티 함수(encoder, learning rate scheduler, etc)

## Installation







