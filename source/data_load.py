from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import matplotlib.image
import numpy as np
import cv2


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


class Normalize(object):
    """ Convert image to grayscale and normalize the color range to [0,1] """
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        image_copy = np.copy(image)
        keypoints_copy = np.copy(keypoints)
        # Convert to grayscale
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
        # Scale from [0, 255] to [0, 1]
        image_copy = image_copy / 255.0
        # Scale keypoints to be centered around 0 with a range [-1, 1]
        keypoints_copy = (keypoints_copy - 100) / 50.0
        return {'image': image_copy, 'keypoints': keypoints_copy}


class Rescale(object):
    """ Rescale image in a sample to a given size, if given int as input,
        scales keeping the aspect ratio, simple resize if given tuple. """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        height, width = image.shape[:2]
        if isinstance(self.output_size, int):
            if height > width:
                new_height, new_width = self.output_size * height / width, width
            else:
                new_height, new_width = height, self.output_size * width / height
        else:
            new_height, new_width = self.output_size
        image = cv2.resize(image, (int(new_height), int(new_width)))
        # Scale the key points to match resized image
        keypoints = keypoints * [new_width / width, new_height / height]
        return {'image': image, 'keypoints': keypoints}
