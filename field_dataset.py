from torch.utils.data import Dataset
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_augmentation(original_height, original_width):
    return A.Compose([
        A.GaussianBlur(blur_limit=(3, 11), p=0.25),
        A.MotionBlur(blur_limit=(3, 11), p=0.25),
        A.GlassBlur(sigma=0.1, max_delta=4, iterations=2, p=0.25),
        A.ShiftScaleRotate(shift_limit=30/original_height, scale_limit=0, rotate_limit=0, p=0.5),  # No rotation
        A.RandomCrop(height=original_height*3//4, width=original_height*3//4, p=0.5), 
        A.Resize(height=original_height, width=original_width),  # Resize back to original dimensions
    ])

class FieldDataset(Dataset):
    """
    A custom Dataset class for handling field data in PyTorch,
    designed to work with datasets where each item has image data and a corresponding laser power value,
    alongside multiple labels for multi-output models.
    
    This version of the class assumes data loading on a specific device (e.g., CPU or GPU).
    """
    
    def __init__(self, data: np.ndarray, power: np.ndarray, power_lab: np.ndarray, n2: np.ndarray, training: bool, device=torch.device("cpu")):
        """
        Initializes the FieldDataset instance with field data, laser power values, and multiple sets of labels.

        Parameters:
            data (np.ndarray): The dataset containing the field data, expected to be in the shape 
                               of [num_samples, num_channels, height, width].
            power (np.ndarray): A one-dimensional array of laser power values with shape [num_samples].
            labels_list (list): A list of np.ndarray, where each array represents a set of labels 
                                for a different output of the model. Each array in the list should
                                have shape [num_samples,].
            device (torch.device): The device on which the tensors will be stored.
        """
        self.device = device
        self.training = training
        self.power_label = torch.from_numpy(power_lab).long().to(self.device)  # Convert power values to tensor and move to device
        self.power = torch.from_numpy(power).float().to(self.device)  # Convert power values to tensor and move to device
        self.n2 = torch.from_numpy(n2).long().to(self.device)
        self.augmentation = get_augmentation(data.shape[-1], data.shape[-1])

        self.data = torch.from_numpy(data).float().to(self.device)

        if self.training:
            # Split channels for amplitude and phase
            for i in range(data.shape[0]):
                if self.device.type == 'cpu':
                    channels = torch.from_numpy(data)[i,:, :, :].permute(1, 2, 0).numpy()  # Replace with your actual amplitude channels
                    # Apply augmentations
                    augmented = self.augmentation(image=channels)['image']
                    self.data[i,:,:,:] = torch.from_numpy(augmented).float().permute(2, 0, 1).to(self.device)
                else:
                    channels = torch.from_numpy(data)[i,:, :, :].permute(1, 2, 0).cpu().numpy()  # Replace with your actual amplitude channels
                    # Apply augmentations
                    augmented = self.augmentation(image=channels)['image']
                    self.data[i,:,:,:] = torch.from_numpy(augmented).float().permute(2, 0, 1).to(self.device)
    def __len__(self) -> int:
        """
        Returns the number of items in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves the data, its corresponding laser power value, and set of labels at the specified index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the data tensor, power value tensor, and a list of label tensors for the given index.
        """
        data_item = self.data[idx,:,:,:]
        power_value =self.power[idx]
        power_labels =self.power_label[idx]
        labels = self.n2[idx]

        return  data_item, power_value, power_labels, labels
