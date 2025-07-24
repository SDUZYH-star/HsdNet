import torch
from torch import nn

# Squeeze-and-Excitation (SE) attention module
class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.PReLU()  # Parametric ReLU activation
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid for attention weights

    def forward(self, x):
        module_input = x  # Save input for residual connection
        x = self.avg_pool(x)  # Global context vector
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Channel-wise attention weights
        return module_input * x  # Apply attention to input features

# Residual block with SE attention
class block(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=True, strides=2):
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv1d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        # Second convolutional layer
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1, stride=1)
        # 1x1 convolution for residual connection (optional)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
        # SE attention module
        self.se = SEModule(num_channels, 16)  # Reduction ratio=16
        self.relu = nn.PReLU()  # Parametric ReLU activation

    def forward(self, X):
        # First convolution + BN + activation
        Y = self.relu(self.bn1(self.conv1(X)))
        # Second convolution + BN
        Y = self.bn2(self.conv2(Y))
        # Process residual connection
        if self.conv3:
            X = self.conv3(X)  # Transform residual
        # Apply SE attention
        Y = self.se(Y)
        # Residual connection
        Y += X
        return Y

# Main HSDNet architecture
class hsdnet(nn.Module):  
    def __init__(self):  
        super(hsdnet, self).__init__()
        # Initial convolution
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)  # Output: 32x4096
        self.bn = nn.BatchNorm1d(32)  # Batch normalization
        # Residual blocks with downsampling
        self.b1 = block(32, 64)    # Output: 64x2048
        self.b2 = block(64, 128)   # Output: 128x1024
        self.b3 = block(128, 256)  # Output: 256x512
        # GRU layer for sequence processing
        self.gru = nn.GRU(512, 512, batch_first=True, bidirectional=False)  
        self.relu = nn.PReLU()  # Activation function
        self.dropout = nn.Dropout(0.5)  # Regularization
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)  # Final classification layer

    def forward(self, x):  
        # Reshape input to (batch, channels, length)
        x = x.view(x.size(0), 1, 4096)
        
        # Initial processing
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        
        # Residual blocks
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        
        # GRU processing
        # Input shape: (batch, channels, length)
        x, _ = self.gru(x)
        # Take last timestep output
        x = x[:, -1, :]
        
        # Classifier
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x