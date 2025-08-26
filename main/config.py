import os
import numpy as np
import pandas as pd
import torch

np.set_printoptions(precision=4, suppress=True)
pd.set_option('display.max_columns', None)

BASE_DIR = "D:/xxxx/main/1_zhu/LLD/"

CASCADE_STAGES = [
    {
        "NAME": "64_32",
        "TRAIN_WELLS": [
            {"LR_PATH": f"{BASE_DIR}2/2H_64.csv", "HR_PATH": f"{BASE_DIR}2/2H_32.csv", "NAME": "Well1"},
            {"LR_PATH": f"{BASE_DIR}4/4H_64.csv", "HR_PATH": f"{BASE_DIR}4/4H_32.csv", "NAME": "Well2"},
            {"LR_PATH": f"{BASE_DIR}3/3H_64.csv", "HR_PATH": f"{BASE_DIR}3/3H_32.csv", "NAME": "Well3"}
        ],
        "TEST_WELL": {
            "LR_PATH": f"{BASE_DIR}1/1H_64.csv",
            "HR_PATH": f"{BASE_DIR}1/1H_32.csv",
            "NAME": "TestWell"
        },
        "OUTPUT_PATH": f"{BASE_DIR}1/test_LLD_64_32_pred.csv",
        "METRICS_PATH": f"{BASE_DIR}1/LLD_64_32_metrics.csv"
    },
    {
        "NAME": "32_16",
        "TRAIN_WELLS": [
            {"LR_PATH": f"{BASE_DIR}2/2H_32.csv", "HR_PATH": f"{BASE_DIR}2/2H_16.csv", "NAME": "Well1"},
            {"LR_PATH": f"{BASE_DIR}4/4H_32.csv", "HR_PATH": f"{BASE_DIR}4/4H_16.csv", "NAME": "Well2"},
            {"LR_PATH": f"{BASE_DIR}3/3H_32.csv", "HR_PATH": f"{BASE_DIR}3/3H_16.csv", "NAME": "Well3"}
        ],
        "TEST_WELL": {
            "LR_PATH": "$PREV_OUTPUT",  # Placeholder to be replaced with previous stage output
            "HR_PATH": f"{BASE_DIR}1/1H_16.csv",
            "NAME": "TestWell"
        },
        "OUTPUT_PATH": f"{BASE_DIR}1/test_LLD_32_16_pred.csv",
        "METRICS_PATH": f"{BASE_DIR}1/LLD_32_16_metrics.csv"
    },
    {
        "NAME": "16_8",
        "TRAIN_WELLS": [
            {"LR_PATH": f"{BASE_DIR}2/2H_16.csv", "HR_PATH": f"{BASE_DIR}2/2H_8.csv", "NAME": "Well1"},
            {"LR_PATH": f"{BASE_DIR}4/4H_16.csv", "HR_PATH": f"{BASE_DIR}4/4H_8.csv", "NAME": "Well2"},
            {"LR_PATH": f"{BASE_DIR}3/3H_16.csv", "HR_PATH": f"{BASE_DIR}3/3H_8.csv", "NAME": "Well3"}
        ],
        "TEST_WELL": {
            "LR_PATH": "$PREV_OUTPUT",
            "HR_PATH": f"{BASE_DIR}1/1H_8.csv",
            "NAME": "TestWell"
        },
        "OUTPUT_PATH": f"{BASE_DIR}1/test_LLD_16_8_pred.csv",
        "METRICS_PATH": f"{BASE_DIR}1/LLD_16_8_metrics.csv"
    },
    {
        "NAME": "8_4",
        "TRAIN_WELLS": [
            {"LR_PATH": f"{BASE_DIR}2/2H_8.csv", "HR_PATH": f"{BASE_DIR}2/2H_4.csv", "NAME": "Well1"},
            {"LR_PATH": f"{BASE_DIR}4/4H_8.csv", "HR_PATH": f"{BASE_DIR}4/4H_4.csv", "NAME": "Well2"},
            {"LR_PATH": f"{BASE_DIR}3/3H_8.csv", "HR_PATH": f"{BASE_DIR}3/3H_4.csv", "NAME": "Well3"}
        ],
        "TEST_WELL": {
            "LR_PATH": "$PREV_OUTPUT",
            "HR_PATH": f"{BASE_DIR}1/1H_4.csv",
            "NAME": "TestWell"
        },
        "OUTPUT_PATH": f"{BASE_DIR}1/test_LLD_8_4_pred.csv",
        "METRICS_PATH": f"{BASE_DIR}1/LLD_8_4_metrics.csv"
    },
    {
        "NAME": "4_2",
        "TRAIN_WELLS": [
            {"LR_PATH": f"{BASE_DIR}2/2H_4.csv", "HR_PATH": f"{BASE_DIR}2/2H_2.csv", "NAME": "Well1"},
            {"LR_PATH": f"{BASE_DIR}4/4H_4.csv", "HR_PATH": f"{BASE_DIR}4/4H_2.csv", "NAME": "Well2"},
            {"LR_PATH": f"{BASE_DIR}3/3H_4.csv", "HR_PATH": f"{BASE_DIR}3/3H_2.csv", "NAME": "Well3"}
        ],
        "TEST_WELL": {
            "LR_PATH": "$PREV_OUTPUT",
            "HR_PATH": f"{BASE_DIR}1/1H_2.csv",
            "NAME": "TestWell"
        },
        "OUTPUT_PATH": f"{BASE_DIR}1/test_LLD_4_2_pred.csv",
        "METRICS_PATH": f"{BASE_DIR}1/LLD_4_2_metrics.csv"
    },
    {
        "NAME": "2_1",
        "TRAIN_WELLS": [
            {"LR_PATH": f"{BASE_DIR}2/2H_2.csv", "HR_PATH": f"{BASE_DIR}2/2.csv", "NAME": "Well1"},
            {"LR_PATH": f"{BASE_DIR}4/4H_2.csv", "HR_PATH": f"{BASE_DIR}4/4.csv", "NAME": "Well2"},
            {"LR_PATH": f"{BASE_DIR}3/3H_2.csv", "HR_PATH": f"{BASE_DIR}3/3.csv", "NAME": "Well3"}
        ],
        "TEST_WELL": {
            "LR_PATH": "$PREV_OUTPUT",
            "HR_PATH": f"{BASE_DIR}1/1.csv",
            "NAME": "TestWell"
        },
        "OUTPUT_PATH": f"{BASE_DIR}1/test_LLD_2_1_pred.csv",
        "METRICS_PATH": f"{BASE_DIR}1/LLD_2_1_metrics.csv"
    }
]

# Training hyperparameters
CURVE_NAME = "LLD"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 10

LOSS_WEIGHTS = {
    'mse': 1.0,
    'mae': 1.0,
    'ssim': 0.1,
    'gradient': 0.5
}

HF_ENHANCEMENT = {
    'noise_intensity': 0.08,
    'hf_boost_factor': 2.5
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
np.random.seed(42)
