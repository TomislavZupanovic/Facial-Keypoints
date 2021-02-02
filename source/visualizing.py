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

    @staticmethod
    def visualize_output(images, pred_keypoints, keypoints=None):
        figure, ax = plt.subplots(1, 5, figsize=(15, 6))
        figure.suptitle('Model predictions', fontsize=15)
        for i in range(5):
            # Un-transform image
            image = images[i].data
            image = image.numpy()
            image = np.transpose(image, (1, 2, 0))
            print(f'Image: {image.shape}')
            # Un-transform key points
            pred = pred_keypoints[i].data
            pred = pred.numpy()
            pred = pred * 50.0 + 100
            print(f'Prediction: {pred.shape}')
            # Plotting
            ax[i].imshow(np.squeeze(image), cmap='gray')
            ax[i].scatter(pred[:, 0], pred[:, 1], s=5, marker='.', c='red')
            if keypoints is not None:
                real = keypoints[i].data
                real = real.numpy()
                real = real * 50.0 + 100
                ax[i].scatter(real[:, 0], real[:, 1], s=5, marker='.', c='yellow')
                print(f'Real: {real.shape}')
            ax[i].axis('off')
        plt.show()
