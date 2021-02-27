import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import cv2
import os


class CNN(nn.Module):
    """ Convolution Neural Network for key points predictions """
    def __init__(self):
        super(CNN, self).__init__()
        # Input image: 1 x 224 x 224
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2, padding=1)
        # 16 x 109 x 109
        self.pool = nn.MaxPool2d(2, 2)
        self.batch1 = nn.BatchNorm2d(16)
        # 16 x 54 x 54
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=1)
        # 32 x 24 x 24
        # After MaxPool: 32 x 12 x 12
        self.batch2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.batch3 = nn.BatchNorm2d(64)
        # 64 x 10 x 10
        # After MaxPool: 64 x 5 x 5
        self.linear = nn.Linear(64 * 5 * 5, 512)
        self.dropout = nn.Dropout(p=0.3)
        # Output 136, 2 for each 68 key point (x, y) pairs
        self.output = nn.Linear(512, 136)
        """ Specify loss function and optimizer """
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=0.001)

    def forward(self, x):
        """ Defines feed forward pass """
        # Calculate output from convolution layers
        x = self.batch1(self.pool(f.relu(self.conv1(x))))
        x = self.batch2(self.pool(f.relu(self.conv2(x))))
        x = self.batch3(self.pool(f.relu(self.conv3(x))))
        # Flatten tensor to vector for linear layer
        x = x.view(-1, 64 * 5 * 5)
        x = f.relu(self.linear(x))
        # Apply dropout
        x = self.dropout(x)
        x = self.output(x)
        return x

    def fit(self, train_loader, epochs):
        """ Trains the CNN on the training data with given number of epochs """
        print('\nStarting Training...\n')
        self.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_num, data in enumerate(train_loader):
                self.optimizer.zero_grad()
                # Get images and their key points
                images = data['image'].type(torch.FloatTensor)
                keypoints = data['keypoints'].type(torch.FloatTensor)
                # Flatten points
                keypoints = keypoints.view(keypoints.size(0), -1)
                output = self.forward(images)
                loss = self.loss_function(output, keypoints)
                # Calculate gradients
                loss.backward()
                # Perform back propagation
                self.optimizer.step()
                running_loss += loss.item()
                if batch_num % 10 == 0:
                    print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_num+1}, Avg. Loss: {running_loss / 10}')
                    running_loss = 0.0
        print('\nFinished Training!')

    def save_model(self):
        """ Saves the model in directory """
        directory = 'saved_models/'
        num_of_files = len(os.listdir(directory))
        model_name = f'Model-{num_of_files}.pt'
        torch.save(self.state_dict(), directory+model_name)

    def load_model(self, model_num=None):
        """ Load the saved model from directory, if model number not specified,
            loads the last one trained """
        directory = 'source/saved_models/'
        if model_num is None:
            num = len(os.listdir(directory))
            model_name = f'Model-{num}.pt'
        else:
            model_name = f'Model-{model_num}.pt'
        self.load_state_dict(torch.load(directory + model_name))
        self.eval()

    def check_predict(self, data_loader):
        """ Predicts key points on sample of given data loader """
        sample = next(iter(data_loader))
        images, keypoints = sample['image'], sample['keypoints']
        images = images.type(torch.FloatTensor)
        pred_keypoints = self.forward(images)
        # Reshape to batch_size x 68 x 2
        pred_keypoints = pred_keypoints.view(pred_keypoints.size()[0], 68, -1)
        return images, pred_keypoints, keypoints


class Detector(object):
    """ Combines Haar Cascade as face detector and CNN as key point detector """
    def __init__(self, cnn_version=None):
        self.face_detector = cv2.CascadeClassifier('source/saved_models/face_detector/haarcascade_frontalface_default.xml')
        self.keypoint_detector = CNN()
        self.keypoint_detector.load_model(model_num=cnn_version)

    @staticmethod
    def transform_roi(roi):
        """ Transform faces as key point detector input """
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (224, 224)) / 255.0
        reshaped = resized.reshape(resized.shape[0], resized.shape[1], 1)
        reshaped = reshaped.transpose((2, 0, 1))
        # Convert to Float Tensor
        torch_img = torch.from_numpy(reshaped).type(torch.FloatTensor)
        # Add batch size of 1
        tensor = torch.unsqueeze(torch_img, 0)
        return tensor

    def image_predict(self, image, scale=False):
        """ Predicts faces and facial key points from given image """
        if scale:
            scale_percent = 80  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        image_copy = image.copy()
        faces = self.face_detector.detectMultiScale(image, 1.2, 2)
        for (x, y, w, h) in faces:
            roi = image[y: y+h+20, x: x+w+20]
            model_input = self.transform_roi(roi)
            prediction = self.keypoint_detector.forward(model_input)
            predictions = prediction.view(prediction.size()[0], 68, -1)
            key_points = predictions[0].data.numpy()
            key_points = key_points * 50.0 + 100
            roi_resized = cv2.resize(roi, (224, 224))
            for dot in range(key_points.shape[0]):
                cv2.circle(roi_resized, (key_points[dot, 0], key_points[dot, 1]), 1, (0, 255, 0), -1, cv2.LINE_AA)
            roi_orig = cv2.resize(roi_resized, (roi.shape[1], roi.shape[0]))
            image_copy[y: y+h+20, x: x+w+20] = roi_orig
        cv2.imshow('Key points prediction', image_copy)
        # cv2.waitKey()

    def webcam_predict(self, frame, frame_width, frame_height):
        image = cv2.resize(frame, (224, 224))
        original = image.copy()
        tensor = self.transform_roi(image)
        prediction = self.keypoint_detector.forward(tensor)
        predictions = prediction.view(prediction.size()[0], 68, -1)
        key_points = predictions[0].data.numpy()
        key_points = key_points * 50.0 + 100
        for dot in range(key_points.shape[0]):
            cv2.circle(original, (key_points[dot, 0], key_points[dot, 1]), 1, (0, 255, 0), -1, cv2.LINE_AA)
        original = cv2.resize(original, (frame_width, frame_height))
        cv2.imshow('Keypoints prediction', original)
