# F-LCAM : Frequency Light Convolutional Attention Module for Sound Event Detection

This repository contains the source for a manuscript "F-LCAM : Frequency Light Convolutional Attention Module for Sound Event Detection."

![SED_Overview](https://github.com/user-attachments/assets/d064d6d5-4b37-412d-ae18-5198385a3e98)

A PDF will be available after publication.


## Data Preparation
[DCASE 2022 Task4 Description](https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments#download) 페이지를 참고하여 데이터 셋을 다운로드 받을 수 있습니다. 
다운로드 후, config.yaml 파일에서 데이터셋 경로를 지정해야합니다.

## File Description
├── main.py # 학습, 검증, 테스트를 위한 메인 스크립트

├── configs

│ └── config.yaml # 모델, 데이터셋 및 학습 설정 파일

├── utils

│ ├── data_aug.py # 데이터 증강 기법 

│ ├── dataset.py # strong/weak/unlabeled data 데이터셋 로더

│ ├── evaluation_measures.py # 평가 지표 계산 (PSDS1/PSDS2 , F1_score)

│ ├── settings_CRNN.py # 모델 설정, 데이터 로딩, 로깅 및 저장 기능

│ ├── utils.py # 유틸리티 함수 (인코더, 학습률 스케줄러 등)

│ └── model_FLCAM.py # Attention 기법을 포함한 CRNN 모델 정의

└── exps/ # 모델 체크포인트 및 로그 저장 디렉토리






