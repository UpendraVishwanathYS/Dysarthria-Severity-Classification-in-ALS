import os
import re
import random
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from transformers import AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model

import librosa
import laserembeddings

import importlib

def load_fold_data(fold_name, data_dir, train_xlsx, test_xlsx):
    train_df = pd.read_excel(train_xlsx, sheet_name=fold_name)
    test_df = pd.read_excel(test_xlsx, sheet_name=fold_name)

    def load_tensor_list(file_names):
        return [torch.load(os.path.join(data_dir, f)) for f in file_names]

    train_data = load_tensor_list(train_df['data'])
    train_labels = load_tensor_list(train_df['label'])
    test_data = load_tensor_list(test_df['data'])
    test_labels = load_tensor_list(test_df['label'])

    # Stack if they are stored individually
    train_data_tensor = torch.cat(train_data)
    train_labels_tensor = torch.cat(train_labels)
    test_data_tensor = torch.cat(test_data)
    test_labels_tensor = torch.cat(test_labels)

    return train_data_tensor, test_data_tensor, train_labels_tensor, test_labels_tensor


def load_classification_model(model_type: str, input_shape: int, num_classes: int):
    module = importlib.import_module(model_type)
    if hasattr(module, 'get_model'):
        return module.get_model(input_shape, num_classes)
    else:
        raise ImportError(f"The module '{model_type}.py' must define a 'get_model' function.")
