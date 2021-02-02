from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import pandas as pd
import os
import matplotlib.image
import numpy as np
import cv2
import torch


class KeypointDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=False):
        """
        Args:
            csv_path (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (optional): Optional transform to the sample.
        """
        self.keypoints_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform_bool = transform
        self.transform = transforms.Compose([Rescale(250), RandomCrop(224),
                                             Normalize(), ToTensor()])

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
        keypoints = self.keypoints_frame.iloc[index, 1:].values
        keypoints = keypoints.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': keypoints}
        if self.transform_bool:
            sample = self.transform(sample)
        return sample

    def get_dataloader(self, batch_size):
        """ Uses PyTorch DataLoaders to load and shuffle data in batches for training model,
            returns training or testing DataLoader """
        data_loader = DataLoader(self, batch_size=batch_size, shuffle=True)
        return data_loader

    def check_data(self):
        """ Checks for proper data format """
        if self.transform_bool:
            data_size = len(self)
            sample = self[np.random.randint(0, data_size - 1)]
            image_size, keypoint_size = sample['image'].size(), sample['keypoints'].size()
            assert image_size == (1, 224, 224), 'Image size or channel is not as expected.'
            assert keypoint_size == (68, 2), 'Keypoints size is not as expected.'
            print('Image and keypoint sizes are correct.')
        else:
            print('Check data only when applying transformations.')


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
                new_height, new_width = self.output_size * height / width, self.output_size
            else:
                new_height, new_width = self.output_size, self.output_size * width / height
        else:
            new_height, new_width = self.output_size
        image = cv2.resize(image, (int(new_height), int(new_width)))
        # Scale the key points to match resized image
        keypoints = keypoints * [int(new_width) / width, int(new_height) / height]
        return {'image': image, 'keypoints': keypoints}


class RandomCrop(object):
    """ Randomly crop image in a sample, if given int, square crop is made otherwise tuple """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        height, width = image.shape[:2]
        new_height, new_width = self.output_size
        # Get random positions in height and width with given crop size
        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, width - new_width)
        # Crop image
        image = image[top: top + new_height, left: left + new_width]
        # Scale key points
        keypoints = keypoints - [left, top]
        return {'image': image, 'keypoints': keypoints}


class ToTensor(object):
    """ Convert image and key points numpy arrays to Tensors. """
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # If image has no grayscale channel, add it
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
        # Reshape color dimensions to torch Tensor: [channel, height, width]
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'keypoints': torch.from_numpy(keypoints)}
