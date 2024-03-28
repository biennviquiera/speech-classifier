import torch.nn as nn
import torch.nn.functional as F

class CNNWithGAP(nn.Module):
    def __init__(self, num_channels, output_size=2):
        super(CNNWithGAP, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, stride=1, padding=1)
        
        # Global Average Pooling

        # Final classification layer
        self.fc = nn.Linear(num_channels * 2, output_size)

    def forward(self, x):
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Reduces each feature map to 1x1

        # Flatten the output for the linear layer
        x = x.view(x.size(0), -1)  # Flatten all dimensions except the batch dimension

        # Classification layer
        x = self.fc(x)
        return x
