from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomAffine, ColorJitter,  Normalize, ToTensor, RandomRotation
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import numpy as np
import sys


import logging


logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Class for the utils. In this case, we are using the MobileNetV2 utils from Keras
# class CNN(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""
#
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

class CNN(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_shape,
                          32,
                          kernel_size=5,
                          padding=0,
                          stride=1,
                          bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32,
                          64,
                          kernel_size=5,
                          padding=0,
                          stride=1,
                          bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.fc1 = nn.Sequential(
                nn.Linear(mid_dim*4, 512),
                nn.ReLU(inplace=True)
            )
            self.fc = nn.Linear(512, num_classes)
        except Exception as e:
            print("CNN")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            out = self.conv1(x)
            out = self.conv2(out)
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = self.fc(out)
            return out
        except Exception as e:
            print("CNN forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

class CNN_3(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN_3, self).__init__()
    #         self.conv1 = nn.Sequential(
    #             nn.Conv2d(input_shape,
    #                       32,
    #                       kernel_size=5,
    #                       padding=0,
    #                       stride=1,
    #                       bias=True),
    #             nn.ReLU(inplace=True),
    #             nn.MaxPool2d(kernel_size=(2, 2))
    #         )
    #         self.conv2 = nn.Sequential(
    #             nn.Conv2d(32,
    #                       64,
    #                       kernel_size=5,
    #                       padding=0,
    #                       stride=1,
    #                       bias=True),
    #             nn.ReLU(inplace=True),
    #             nn.MaxPool2d(kernel_size=(2, 2))
    #         )
    #         self.fc1 = nn.Sequential(
    #             nn.Linear(mid_dim*4, 512),
    #             nn.ReLU(inplace=True)
    #         )
    #         self.fc = nn.Linear(512, num_classes)
    #     except Exception as e:
    #         print("CNN")
    #         print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
    #
    # def forward(self, x):
    #     try:
    #         out = self.conv1(x)
    #         out = self.conv2(out)
    #         out = torch.flatten(out, 1)
    #         out = self.fc1(out)
    #         out = self.fc(out)
    #         return out
    #     except Exception as e:
    #         print("CNN forward")
    #         print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            self.model = torch.nn.Sequential(

                # queda para asl
                # nn.Conv2d(input_shape, 32, kernel_size=3, padding=1),
                # nn.ReLU(),
                # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                # nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
                #
                # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                # nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
                # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                # nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
                #
                # nn.Flatten(),
                # nn.Linear(mid_dim,512),
                # nn.ReLU(),
                # nn.Linear(512, num_classes))

                # nn.Linear(28*28, 392),
                # nn.ReLU(),
                # nn.Dropout(0.5),
                # nn.Linear(392, 196),
                # nn.ReLU(),
                # nn.Linear(196, 98),
                # nn.ReLU(),
                # nn.Dropout(0.3),
                # nn.Linear(98, num_classes)

                torch.nn.Conv2d(in_channels=input_shape, out_channels=32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                # Input = 32 x 32 x 32, Output = 32 x 16 x 16
                torch.nn.MaxPool2d(kernel_size=2),

                # Input = 32 x 16 x 16, Output = 64 x 16 x 16
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                # Input = 64 x 16 x 16, Output = 64 x 8 x 8
                torch.nn.MaxPool2d(kernel_size=2),

                # Input = 64 x 8 x 8, Output = 64 x 8 x 8
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                # Input = 64 x 8 x 8, Output = 64 x 4 x 4
                torch.nn.MaxPool2d(kernel_size=2),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                # Input = 64 x 8 x 8, Output = 64 x 4 x 4
                torch.nn.MaxPool2d(kernel_size=2),

                torch.nn.Flatten(),
                torch.nn.Linear(mid_dim * 4 * 4, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, num_classes)
            )

        except Exception as e:

            print("CNN_3 init")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def forward(self, x):
        try:
            return self.model(x)
        except Exception as e:
            print("CNN_3 forward")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class CNN_3_proto(torch.nn.Module):
    def __init__(self, input_shape, mid_dim=64, num_classes=10):

        try:
            super(CNN_3_proto, self).__init__()

                # queda para asl
                # nn.Conv2d(input_shape, 32, kernel_size=3, padding=1),
                # nn.ReLU(),
                # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                # nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
                #
                # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                # nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
                # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                # nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
                #
                # nn.Flatten(),
                # nn.Linear(mid_dim,512),
                # nn.ReLU(),
                # nn.Linear(512, num_classes))

                # nn.Linear(28*28, 392),
                # nn.ReLU(),
                # nn.Dropout(0.5),
                # nn.Linear(392, 196),
                # nn.ReLU(),
                # nn.Linear(196, 98),
                # nn.ReLU(),
                # nn.Dropout(0.3),
                # nn.Linear(98, num_classes)

            self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=input_shape, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Input = 32 x 32 x 32, Output = 32 x 16 x 16
            torch.nn.MaxPool2d(kernel_size=2),

            # Input = 32 x 16 x 16, Output = 64 x 16 x 16
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Input = 64 x 16 x 16, Output = 64 x 8 x 8
            torch.nn.MaxPool2d(kernel_size=2),

            # Input = 64 x 8 x 8, Output = 64 x 8 x 8
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Input = 64 x 8 x 8, Output = 64 x 4 x 4
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),

            torch.nn.ReLU(),
            # Input = 64 x 8 x 8, Output = 64 x 4 x 4
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(mid_dim * 4 * 4, 512))

            self.fc = torch.nn.Linear(512, num_classes)

        except Exception as e:
            logger.info("CNN_3_proto")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def forward(self, x):
        try:
            proto = self.conv1(x)
            out = self.fc(proto)
            return out, proto
        except Exception as e:
            logger.info("CNN_3_proto")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class CNN_student(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN_student, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_shape,
                          32,
                          kernel_size=3,
                          padding=0,
                          stride=1,
                          bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Flatten(),
                nn.Linear(mid_dim * 4, 512))
            # self.out = nn.Linear(512, num_classes)
            # self.conv1 = nn.Sequential(
            #     nn.Conv2d(input_shape,
            #               32,
            #               kernel_size=3,
            #               padding=0,
            #               stride=1,
            #               bias=True),
            #     nn.ReLU(inplace=True),
            #     nn.MaxPool2d(kernel_size=(2, 2)),
            #     torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
            #     torch.nn.ReLU(),
            #     # Input = 64 x 16 x 16, Output = 64 x 8 x 8
            #     torch.nn.MaxPool2d(kernel_size=2),
            #     nn.Flatten(),
            #     nn.Linear(mid_dim * 4, 512),
            #     nn.ReLU(inplace=True))
            self.out = nn.Linear(512, num_classes)
        except Exception as e:
            logger.info("CNN student")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def forward(self, x):
        try:
            proto = self.conv1(x)
            out = self.out(proto)
            return out, proto
        except Exception as e:
            logger.info("CNN student forward")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class CNNDistillation(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10, dataset='CIFAR10'):
        try:
            self.dataset = dataset
            super(CNNDistillation, self).__init__()
            self.new_client = False
            if self.dataset in ['EMNIST', 'MNIST']:
                # mid_dim = 1568
                mid_dim = 1352 # CNN 1 pad 1
                # mid_dim = 400
            else:
                # mid_dim = 400
                mid_dim = 1800 # cnn student 1 cnn
                # mid_dim = 576 # cnn student 2 cnn
            self.student = CNN_student(input_shape=input_shape, mid_dim=mid_dim, num_classes=num_classes)
            if self.dataset in ['CIFAR10', 'GTSRB']:
                mid_dim = 16
            else:
                mid_dim = 4
            self.teacher = CNN_3_proto(input_shape=input_shape, mid_dim=mid_dim, num_classes=num_classes)
        except Exception as e:
            logger.info("CNNDistillation")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def forward(self, x):
        try:
            out_student, proto_student = self.student(x)
            out_teacher, proto_teacher = self.teacher(x)
            return out_student, proto_student, out_teacher, proto_teacher
        except Exception as e:
            logger.info("CNNDistillation forward")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))