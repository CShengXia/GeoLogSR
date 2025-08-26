import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    x_values = data.iloc[:, 0].values
    y_values = data.iloc[:, 1].values
    return x_values, y_values

def prepare_datasets_for_training(low_res_x, low_res_y, high_res_x, high_res_y, window_size=64, stride=16):

    ratio = len(high_res_y) / len(low_res_y)
    X_train = []
    Y_train = []
    for i in range(0, len(low_res_y) - window_size, stride):
        lr_window = low_res_y[i : i + window_size]
        # Calculate corresponding high-res window indices (assuming roughly 2x resolution)
        hr_start = int(i * ratio)
        hr_end = hr_start + int(window_size * ratio)
        if hr_end <= len(high_res_y):
            hr_window = high_res_y[hr_start:hr_end]
            # Only use the window if it exactly matches the expected size
            if len(hr_window) == int(window_size * ratio):
                X_train.append(lr_window)
                Y_train.append(hr_window)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    return X_train, Y_train

def prepare_datasets_from_multiple_wells(wells_config, window_size=64, stride=16):
    X_train_list = []
    Y_train_list = []
    for well in wells_config:
        lr_x, lr_y = load_data(well["LR_PATH"])
        hr_x, hr_y = load_data(well["HR_PATH"])
        well_X, well_Y = prepare_datasets_for_training(lr_x, lr_y, hr_x, hr_y, window_size=window_size, stride=stride)
        if len(well_X) > 0:
            X_train_list.append(well_X)
            Y_train_list.append(well_Y)
        else:
            alt_X = []
            alt_Y = []
            for i in range(0, len(lr_y) - window_size, stride):
                x_window = lr_y[i : i + window_size]
                x_coords = np.arange(window_size)
                x_upsampled = np.linspace(0, window_size - 1, window_size * 2)
                y_upsampled = np.interp(x_upsampled, x_coords, x_window)
                alt_X.append(x_window)
                alt_Y.append(y_upsampled)
            if len(alt_X) > 0:
                X_train_list.append(np.array(alt_X))
                Y_train_list.append(np.array(alt_Y))
    if X_train_list:
        X_train = np.vstack(X_train_list)
        Y_train = np.vstack(Y_train_list)
    else:
        X_train = np.array([])
        Y_train = np.array([])
    return X_train, Y_train

def prepare_full_dataset(low_res_y, high_res_y=None):
    X = np.expand_dims(np.expand_dims(low_res_y, axis=0), axis=0).astype(np.float32)
    X_tensor = torch.tensor(X)
    if high_res_y is not None:
        Y = np.expand_dims(np.expand_dims(high_res_y, axis=0), axis=0).astype(np.float32)
        Y_tensor = torch.tensor(Y)
        return X_tensor, Y_tensor
    return X_tensor

def enhance_high_freq(signal, factor=1.2, min_width=2, max_width=5):

    result = signal.copy()
    length = len(signal)
    num_spikes = max(3, length // 30)
    for _ in range(num_spikes):
        pos = np.random.randint(0, length)
        width = np.random.randint(min_width, max_width + 1)
        height = np.random.uniform(0.02, 0.1) * factor
        for i in range(max(0, pos - width), min(length, pos + width + 1)):
            dist = abs(i - pos)
            result[i] += height * np.exp(-0.5 * ((dist / width) * 3) ** 2) * np.sign(np.random.randn())
    return result

class WellLogDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]