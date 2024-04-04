# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Experiment with different configurations
class ExperimentCNN(nn.Module):
    def __init__(self):
        super(ExperimentCNN, self).__init__()
        # Modify the convolutional layers, fully connected layers, or any other part of the model
        self.conv1 = nn.Conv2d(1, 16, 3)  # Change number of output channels
        self.conv2 = nn.Conv2d(16, 32, 3)  # Change number of output channels
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32*5*5, 120)  # Adjust input size for the fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.maxpool(self.conv1(x)))
        x = F.relu(self.maxpool(self.conv2(x)))
        x = x.view(-1, 32*5*5)  # Adjust for new feature map size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

