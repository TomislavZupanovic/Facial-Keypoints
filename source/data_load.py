from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import matplotlib.image
import numpy as np


class KeypointDataset(Dataset):
    def __init__(self, csv, root_dir, transform=None):
        """
        Args:
            csv (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (optional): Optional transform to the sample.
        """
        self.keypoints_frame = pd.read_csv(csv)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """ Return the size of CSV """
        return len(self.keypoints_frame)

    def __getitem__(self, index):
        """ Returns the data sample from CSV with given index, returns sample in dictionary
            with image and key points """
        image_name = os.path.join(self.root_dir, self.keypoints_frame.iloc[index, 0])
        image = matplotlib.image.imread(image_name)
        # If image has alpha color channel (channel 4), remove it
        if image.shape[2] == 4:
            image = image[:, :, 0:3]
        keypoints = self.keypoints_frame.iloc[index, 1].as_matrix()
        keypoints = keypoints.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': keypoints}
        if self.transform:
            sample = self.transform(sample)
        return sample
