# Deep Learning-Based Facial Expression Recognition for Hijabi and Non-Hijabi Faces

> 📄 Paper accepted at the **3rd International Conference on Emerging Trends and Applications in Artificial Intelligence (ICETAI 2026)**, Istanbul, Turkey.

## Overview

Facial Expression Recognition (FER) systems often fail to perform equally across demographic groups. This project addresses the **fairness gap** between hijabi and non-hijabi women in FER systems using Deep Learning.

We evaluate fairness-oriented training strategies on a self-collected, balanced dataset of **2,500 facial images** across 5 emotion categories.

**Key finding:** Careful dataset design reduces fairness gaps more effectively than algorithmic interventions alone.

## Authors

Batool Mahashi, Samira Aldamen, **Wajd Almadi**, Shatha Arar, Rasha Obeidat, Farah AlShanik

Jordan University of Science and Technology (JUST), Irbid, Jordan

## Results

| Method | Overall Acc | Fairness Gap | Worst-Group Acc |
|---|---|---|---|
| Baseline (ResNet-18) | 95.48% | 0.48% | 91.89% |
| Balanced Sampling | 93.62% | 2.06% | 78.95% |
| Group-Weighted Loss | 95.74% | 1.02% | 89.47% |
| DANN | 96.81% | 2.16% | 89.19% |

## Repository Structure

```
├── fairness_methods.ipynb         # Baselines + Fairness Methods
├── cross_dataset_evaluation.ipynb # Cross-dataset evaluation
├── gan_augmentation.ipynb         # GAN-based augmentation (exploratory)
└── live_detection.py              # Real-time emotion detection demo
```

## Dataset

Self-collected dataset available on Kaggle:
- 1,250 hijabi women / 1,250 non-hijabi women  
- 5 emotions: Happy, Sad, Angry, Surprised, Neutral

🔗 [Hijabi and Non-Hijabi Facial Expression Dataset](https://www.kaggle.com/datasets/batoolmahashi/hijabi-and-non-hijabi-facial-expression-dataset/data)


## Tech Stack

`Python` `PyTorch` `ResNet-18` `VGG-16` `MobileNetV2` `DenseNet-121` `OpenCV` `Google Colab`
