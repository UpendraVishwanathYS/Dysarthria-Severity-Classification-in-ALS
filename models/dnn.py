import torch
import torch.nn as nn

class SimpleDNN(nn.Module):
    def __init__(self, input_shape, num_classes=2, learning_rate=0.001):
        super(SimpleDNN_drop, self).__init__()
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(input_shape, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 4)
        self.fc5 = nn.Linear(4, num_classes)  

    def forward(self, x):
        x = x.squeeze(1)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.fc1(x)
        if x.shape[0] > 1:
            x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)

        x = nn.ReLU()(self.fc3(x))
        x = nn.ReLU()(self.fc4(x))
        x = self.fc5(x)  # Raw logits (no softmax)

        return x
