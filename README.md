# Segemantation_DeepLabv3
PyTorch 기반 Network에 대해 DeepLabv3 방식으로 Semantic Segmentation을 진행하는 프로젝트 입니다.

## Implementations

- DeepLabv3
- Focal Loss
- Data(Prepare)
- Data Augmentations(Albumentations)
- Muti GPU Training
- (Todo)Cosine Annealing Warmup Restart L.R. Scheduler

## 프로젝트 구조
```
Segmentation_DeepLabv3
├─ .gitignore
├─ __README.md
├─ configs # 학습 시 사용할 하이퍼 파라미터, 데이터셋 설정 등 Configuration을 위한 yaml 파일 경로
├─ dataset # Image Data Generator 모듈
├─ models # Detector, Convolution Module 등 구현
│  ├─ backbone
│  ├─ segmentor
│  ├─ layers
│  └─ loss
├─ module # 학습을 위한 Pytorch Lightning 모듈
├─ onnx_module # onnx 변형을 위한 모듈
├─ train_segmentation.py # segmentor 학습 스크립트
└─ utils

```

## Requirements
`requirements.txt` 파일을 참고하여 Anaconda 환경 설정 (conda install 명령어)  
`PyYaml`  
`PyTorch`  
`Pytorch Lightning`

## Config Train Parameters

기본 설정값은 ./configs/default_settings.yaml에 정의됩니다.  
Train 스크립트 실행 시 입력되는 CFG 파일로 하이퍼파라미터 및 학습 기법을 설정할 수 있습니다.

[default_settings.yaml](./configs/default_settings.yaml)

    // ./configs/*.yaml 파일 수정
    // ex) cls_frostnet -> default_settings 파라미터를 업데이트 해서 사용
    model : 'ResNet'
    dataset_name : Pascal_voc
    classes : 200
    epochs: 1000
    data_path : '/mnt/train/'
    save_dir : './saved'
    workers: 4
    ...

## Train Segmentor

Semantic Segmentation 모델 Train 스크립트 입니다.

    python train_segmenntation.py --cfg configs/seg_resnet.yaml

## Test DeepLabv3
![deeplabv3](./doc/deeplabv3/deeplabv3.PNG)

## TODOs
- Backbone Networks Customizing
- Deployment를 위한 ONNX Conversion Script, Torch Script 추가
- Instance Segmentation 방법 추가
