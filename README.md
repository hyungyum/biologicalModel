# biologicalModel

# 성능 최적화 기법
1.	SWA (Stochastic Weight Averaging)
2.	CutMix 데이터 증강
3.	Label Smoothing
4.	Autoaugment 기법
5.	혼합 정밀도 학습 (Mixed Precision Training)


# Project Directory Structure (example)

```plaintext
classfication/
│
├── data/                                 # Dataset storage
│   ├── cifar-10-batches-py/              # CIFAR-10 dataset files
│   └── ...                               # Other dataset folders
│
├── models/                               # Model architecture files
│   ├── __init__.py                       # Init file for model imports
│   ├── resnet.py                         # ResNet model definition
│   └── toy_network.py                    # A simple toy model for experimentation
│
├── runs/                                 # TensorBoard log files
│   ├── resnet18_cutmix_epoch50_20interval/ # Logs for cutmix experiment
│   ├── resnet18_label_smoothing_epoch50/   # Logs for label smoothing experiment
│   └── ...                               # Logs for other experiments
│
├── ckpt_cutmix/                          # Checkpoints for CutMix model
│   ├── model_050_0.8456.pth              # Best performing CutMix model
│   └── ...                               # Other saved checkpoints
│
├── ckpt_label_smoothing/                 # Checkpoints for Label Smoothing model
│   ├── model_050_0.8603.pth              # Best performing Label Smoothing model
│   └── ...                               # Other saved checkpoints
│
├── scripts/                              # Python training and evaluation scripts
│   ├── cutmix.py                         # CutMix implementation script
│   ├── label_smoothing.py                # Label Smoothing implementation script
│   └── ...                               # Other training scripts
│
├── README.md                             # Project documentation
├── cutmix.py                             # CutMix implementation script     
├── swa.py                                # SWA implementation script           
├── label_smoothing.py                    # Label Smoothing implementation script     
├── label_smoothing_plus_swa.py           # Label Smoothing + SWA implementation script         
└── label_smoothing_plus_swa_autoaugment.py  # AutoAugment + Label Smoothing + SWA implementation script
