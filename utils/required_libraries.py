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
