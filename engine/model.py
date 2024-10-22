import torch
import torch.nn as nn
import torchvision.models as models

class SubModel(nn.Module):
    def __init__(self):
        super(SubModel, self).__init__()
        
        self.net = models.convnext_base(pretrained=True)
        self.net.features[0][0] = nn.Conv2d(2, self.net.features[0][0].out_channels, kernel_size=4, stride=4)
        self.net.classifier = nn.Identity()
        
    def forward(self, x):
        features = self.net(x)
        return features

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        
        self.model = SubModel()

        self.shared_layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        
        self.mean_head = nn.Sequential(
            nn.Linear(512, 3),
            nn.Sigmoid()
        )
        

        self.covariance_head = nn.Sequential(
            nn.Linear(512, 6)
        )

    def forward(self, input):
        features = self.model(input)[:, :, 0, 0]
        features = self.shared_layers(features) 
        
        # Predict means
        mean_predictions = self.mean_head(features)
        
        # Predict covariance matrix elements (variances and covariances)
        cov_predictions = self.covariance_head(features)
        
        return mean_predictions, cov_predictions