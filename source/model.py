import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


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
