import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights

# Define the N2CondNet class for conditional prediction of n2
class N2CondNet(nn.Module):
    def __init__(self, feature_dim=512):
        super(N2CondNet, self).__init__()
        
        # Embedding layers for Isat and alpha
        self.isat_embedding = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.alpha_embedding = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        # Main network for n2 prediction, conditioned on Isat and alpha embeddings
        self.n2_net = nn.Sequential(
            nn.Linear(feature_dim + 2*512, 1024),  # Combined dimension of image features and embeddings
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, isat, alpha):
        # Pass Isat and alpha through their respective embedding layers
        isat_embedded = self.isat_embedding(isat)
        alpha_embedded = self.alpha_embedding(alpha)
        
        # Concatenate image features with the embedded Isat and alpha
        combined = torch.cat((features, isat_embedded, alpha_embedded), dim=1)
        
        # Pass the combined representation through the main network for n2 prediction
        n2 = self.n2_net(combined)
        return n2


# Define the feature extraction model
class SubModel(nn.Module):
    def __init__(self):
        super(SubModel, self).__init__()
        
        # Load pre-trained ConvNeXt model and modify it for two-channel input
        self.net = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.net.features[0][0] = nn.Conv2d(2, self.net.features[0][0].out_channels, kernel_size=4, stride=4)
        self.net.classifier = nn.Identity()  # Remove the final classification layer
        
    def forward(self, x):
        features = self.net(x)
        return features


# Define the full model with shared feature extraction and heads for n2, Isat, and alpha
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        
        # Feature extraction model
        self.model = SubModel()

        # Shared fully connected layers for feature processing
        self.shared_layers = nn.Sequential(
            nn.Linear(768, 2048),
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
        
        # Independent heads for Isat and alpha
        self.isat_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # CondNet for conditioned prediction of n2
        self.n2_function = N2CondNet(feature_dim=512)
        
        # Covariance head for variances and covariances between n2, Isat, and alpha
        self.covariance_head = nn.Linear(512, 6)

    def forward(self, input):
        # Feature extraction from input image

        features = self.model(input)
        features = features[:, :, 0, 0]
        features = self.shared_layers(features)
        
        # Predict Isat and alpha independently
        isat = self.isat_head(features)
        alpha = self.alpha_head(features)

        # Predict n2 using the conditional network
        n2 = self.n2_function(features, isat, alpha)
        
        # Combine predictions into mean_predictions tensor
        mean_predictions = torch.cat((n2, isat, alpha), dim=1)
        
        # Predict covariance matrix elements (variances and covariances)
        cov_predictions = self.covariance_head(features)
        
        return mean_predictions, cov_predictions