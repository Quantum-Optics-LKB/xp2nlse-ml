import torch.nn as nn
import torchvision.models as models

class SubModel(nn.Module):
    def __init__(self):
        super(SubModel, self).__init__()
        
        # Load ConvNeXt-Base with pretrained weights set to False
        self.net = models.convnext_base(pretrained=False)
        
        # Modify the first convolutional layer to accept 2 input channels
        # By default, ConvNeXt-Base has 3 input channels, we'll modify it to accept 2.
        self.net.features[0][0] = nn.Conv2d(
            in_channels=2,    # Change input channels to 2
            out_channels=128,  # ConvNeXt-Base starts with 96 output channels
            kernel_size=4,
            stride=4,
            padding=0,
            bias=False
        )
        
        # Initialize the new convolutional weights properly using Kaiming Normal
        nn.init.kaiming_normal_(self.net.features[0][0].weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        # Forward pass through ConvNeXt
        features = self.net(x)  # Output shape will be [Batch Size, 1024]
        return features

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        
        # Use the modified ConvNeXt-Base as the feature extractor
        self.model = SubModel()
        
        # Define the fully connected layers to predict the 3 output parameters
        self.fc = nn.Sequential(
        nn.Linear(1000, 512),
        nn.SiLU(),  # Replace ReLU with SiLU
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128), 
        nn.SiLU(),
        nn.Linear(128, 64),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 3)
    )
    
    def forward(self, input):
        # Extract features using ConvNeXt
        features = self.model(input)
        # Pass the features through the fully connected layers
        output = self.fc(features)
        return output
