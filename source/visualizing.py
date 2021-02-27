import numpy as np
import matplotlib.pyplot as plt


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
    def plot_lines(idx, pred, ax):
        """ Plots lines between key points """
        points = [0, 17, 22, 27, 31, 36, 42, 48, 60]
        for p in range(len(points) - 1):
            j = p + 1
            ax[idx].plot(pred[points[p]:points[j], 0], pred[points[p]:points[j], 1], linestyle='solid', c='red')
        # Extra line plots to close holes in key points
        ax[idx].plot([pred[36, 0], pred[41, 0]], [pred[36, 1], pred[41, 1]], linestyle='solid', c='red')
        ax[idx].plot([pred[42, 0], pred[47, 0]], [pred[42, 1], pred[47, 1]], linestyle='solid', c='red')
        ax[idx].plot([pred[48, 0], pred[59, 0]], [pred[48, 1], pred[59, 1]], linestyle='solid', c='red')

    @staticmethod
    def visualize_output(images, pred_keypoints, lines=False, keypoints=None):
        """ Visualizes the model predicted key points on given images """
        # Define figure and subplots
        figure, ax = plt.subplots(1, 4, figsize=(10, 5))
        figure.suptitle('Model predictions', fontsize=15)
        for i in range(4):
            # Un-transform image
            image = images[i].data
            image = image.numpy()
            image = np.transpose(image, (1, 2, 0))
            # Un-transform key points
            pred = pred_keypoints[i].data
            pred = pred.numpy()
            pred = pred * 50.0 + 100
            # Plotting
            ax[i].imshow(np.squeeze(image), cmap='gray')
            if not lines:
                ax[i].scatter(pred[:, 0], pred[:, 1], s=5, marker='.', c='red')
            elif lines:
                Visualizer.plot_lines(i, pred, ax)
            if keypoints is not None:
                real = keypoints[i]
                real = real * 50.0 + 100
                ax[i].scatter(real[:, 0], real[:, 1], s=1, marker='.', c='yellow')
            ax[i].axis('off')
        plt.show()
