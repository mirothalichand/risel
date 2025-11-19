# RISEL: Reinforcement Learning–Integrated Stacked Ensemble for Landslide Forecasting

This repository contains the full implementation of the RISEL framework proposed in the manuscript:
"Reinforcement Learning-Integrated Stacked Ensemble Learning for Real-Time Landslide Movement Prediction."

## Features
- Four RL base learners: DQN, A2C, ACM, PPO
- LightGBM, CatBoost, HistGradientBoosting, Logistic Lasso, GBM meta-learners
- 5-fold cross-validation and temporal train–test split
- Reward shaping for class imbalance
- Ablation studies (1/2/3/4 RL models)
- Sensor preprocessing & lag feature creation
- Inference scripts

## Repository Contents
- `/src/` – full source code
- `/RISEL_model.ipynb` – main training + evaluation pipeline
- `/sample data` – anonymized sample dataset

## Requirements
Python 3.8+  
Torch, LightGBM, CatBoost, Scikit-learn, Pandas, NumPy

## Usage
```bash
python src.py

