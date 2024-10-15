import torch.nn as nn
import torchvision.models as models

class SubModel(nn.Module):
    def __init__(self):
        super(SubModel, self).__init__()
        
        self.net = models.efficientnet_b4()
        
        # Modify the first convolutional layer to accept 2 input channels
        # By default, EfficientNet-B4 has 3 input channels, we'll modify it to accept 2.
        self.net.features[0][0] = nn.Conv2d(
            in_channels=2,    # Change input channels to 2
            out_channels=48,  # Keep the original output channels as is
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        
        # Initialize the new convolutional weights properly using Kaiming Normal
        nn.init.kaiming_normal_(self.net.features[0][0].weight, mode='fan_out', nonlinearity='relu')
        
        # Replace the classifier to output 1024 features
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1792, out_features=1024)  # EfficientNet-B4 has 1792 features
        )
        
    def forward(self, x):
        # Forward pass through EfficientNet
        features = self.net(x)  # Output shape: [Batch Size, 1024]
        return features

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        
        # Use the modified EfficientNet as the feature extractor
        self.model = SubModel()
        
        # Define the fully connected layers to predict the 3 output parameters
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3),  # Output 3 values
        )
    
    def forward(self, input):
        # Extract features using EfficientNet
        features = self.model(input)
        # Pass the features through the fully connected layers
        output = self.fc(features)
        return output
