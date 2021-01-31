import numpy as np
import matplotlib.pyplot as plt
import torch


class Visualizer:
    def __init__(self, data):
        self.data = data
        self.dataloader = data.get_dataloader(batch_size=32)

    def data_sample(self):
        """ Shows sample images of data set """
        # TODO: Fix plt.imshow()
        for i in range(4):
            figure = plt.figure(figsize=(10, 5))
            random_num = np.random.randint(0, len(self.data))
            sample = self.data[random_num]
            axis = plt.subplot(4, 1, i+1)
            image, keypoints = sample['image'], sample['keypoints']
            plt.imshow(image)
            plt.scatter(keypoints[:, 0], keypoints[:, 1], s=25, marker='.', c='yellow')
            figure.axes('off')
            plt.show()

