import numpy as np
import matplotlib.pyplot as plt
import torch


class Visualizer:
    def __init__(self, data):
        self.data = data
        self.dataloader = data.get_dataloader(batch_size=32)

    def data_sample(self):
        """ Shows sample images of data set """
        figure, ax = plt.subplots(1, 4, figsize=(15, 6))
        figure.suptitle('Data sample images with key points', fontsize=15)
        for i in range(4):
            random_num = np.random.randint(0, len(self.data))
            sample = self.data[random_num]
            image, keypoints = sample['image'], sample['keypoints']
            ax[i].imshow(image)
            ax[i].scatter(keypoints[:, 0], keypoints[:, 1], s=5, marker='.', c='yellow')
            ax[i].axis('off')
        plt.show()

