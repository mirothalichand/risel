# RISEL: Reinforcement Learning–Integrated Stacked Ensemble for Landslide Forecasting

This repository contains the full implementation of the RISEL framework proposed in the manuscript:
"Reinforcement Learning-Integrated Stacked Ensemble Learning for Real-Time Landslide Movement Prediction."

## Features
- Four RL base learners: DQN, A2C, ACM, PPO
- LightGBM, CatBoost, HistGradientBoosting, Logistic Lasso, GBM meta-learners
- 5-fold cross-validation and temporal train–test split
- Reward shaping for class imbalance
- Ablation studies (1/2/3/4 RL models)
- Cross-site generalizability experiments
- Statistical significance testing (paired t-tests)
- Sensor preprocessing & lag feature creation
- Inference scripts

## Repository Contents
- `/src/` – full source code
- `/notebooks/RISEL_model.ipynb` – main training + evaluation pipeline
- `/data/` – anonymized sample dataset
- `/configs/` – hyperparameter & reward configuration files
- `/results/` – example outputs (confusion matrices, metrics)
- `/docs/` – manuscript-related documentation

## Requirements
Python 3.8+  
Torch, LightGBM, CatBoost, Scikit-learn, Pandas, NumPy

## Usage
```bash
python train_risel.py
python evaluate_risel.py
