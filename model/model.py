import tensorflow as tf
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from tensorflow.python.ops.metrics_impl import accuracy
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomAffine, ColorJitter,  Normalize, ToTensor, RandomRotation
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import numpy as np
import sys


import logging


logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Class for the model. In this case, we are using the MobileNetV2 model from Keras
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
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            return self.model(x)
        except Exception as e:
            print("CNN_3 forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

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
            print("CNN_3_proto")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            proto = self.conv1(x)
            out = self.fc(proto)
            return out, proto
        except Exception as e:
            print("CNN_3_proto")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

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
                nn.Linear(mid_dim * 4, 512),
                nn.ReLU(inplace=True))
            self.out = nn.Linear(512, num_classes)
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
            print("CNN student")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            proto = self.conv1(x)
            out = self.out(proto)
            return out, proto
        except Exception as e:
            print("CNN student forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

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
            print("CNNDistillation")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            out_student, proto_student = self.student(x)
            out_teacher, proto_teacher = self.teacher(x)
            return out_student, proto_student, out_teacher, proto_teacher
        except Exception as e:
            print("CNNDistillation forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)



def load_model(model_name, dataset, strategy):
    if model_name == 'CNN':
        if dataset in ['MNIST']:
            input_shape = 1
            mid_dim = 256
            # mid_dim = 4
            num_classes = 10
            logging.info("""leu mnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
        elif dataset in ['EMNIST']:
            input_shape = 1
            mid_dim = 256
            # mid_dim = 4
            num_classes = 47
            logging.info("""leu emnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
        elif dataset in ['GTSRB']:
            input_shape = 1
            mid_dim = 36
            # mid_dim = 16
            num_classes = 43
            logging.info("""leu gtsrb com {} {} {}""".format(input_shape, mid_dim, num_classes))
        else:
            input_shape = 3
            mid_dim = 400
            # mid_dim = 16
            num_classes = 10
    elif model_name == 'CNN_3':
        if dataset in ['MNIST']:
            input_shape = 1
            # mid_dim = 256
            mid_dim = 4
            num_classes = 10
            logging.info("""leu mnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
        elif dataset in ['EMNIST']:
            input_shape = 1
            # mid_dim = 256
            mid_dim = 4
            num_classes = 47
            logging.info("""leu emnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
        elif dataset in ['GTSRB']:
            input_shape = 3
            # mid_dim = 36
            mid_dim = 16
            num_classes = 43
            logging.info("""leu gtsrb com {} {} {}""".format(input_shape, mid_dim, num_classes))
        else:
            input_shape = 3
            # mid_dim = 400
            mid_dim = 16
            num_classes = 10
            logging.info("""leu cifar com {} {} {}""".format(input_shape, mid_dim, num_classes))
        return CNN_3(input_shape=input_shape, num_classes=num_classes, mid_dim=mid_dim)


fds = None

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def get_weights_fedkd(net):
    return [val.cpu().numpy() for _, val in net.student.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def set_weights_fedkd(net, parameters):
    params_dict = zip(net.student.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.student.load_state_dict(state_dict, strict=True)


def load_data(dataset_name: str, alpha: float, partition_id: int, num_partitions: int, batch_size: int,
              data_sampling_percentage: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    logger.info(
        """Loading {} data.""".format(dataset_name, partition_id, num_partitions, batch_size, data_sampling_percentage))
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label",

                                           alpha=alpha, min_partition_size=10,

                                           self_balancing=True)
        fds = FederatedDataset(
            dataset={"EMNIST": "claudiogsc/emnist_balanced", "CIFAR10": "uoft-cs/cifar10", "MNIST": "ylecun/mnist", "GTSRB": "tanganke/gtsrb"}[dataset_name],
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    test_size = 1 - data_sampling_percentage
    partition_train_test = partition.train_test_split(test_size=test_size, seed=42)

    pytorch_transforms = {"CIFAR10": Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "MNIST": Compose([ToTensor(), RandomRotation(10),
                                           Normalize([0.5], [0.5])]),
        "EMNIST": Compose([ToTensor(), RandomRotation(10),
                          Normalize([0.5], [0.5])]),
        "GTSRB": Compose(
                    [

                        Resize((32, 32)),
                        RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                        RandomRotation(10),  # Rotates the image to a specified angel
                        RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                        # Performs actions like zooms, change shear angles.
                        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                        ToTensor(),
                        Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                    ]
                )
    }[dataset_name]

    # import torchvision.datasets as datasets
    # datasets.EMNIST
    key = {"CIFAR10": "img", "MNIST": "image", "EMNIST": "image", "GTSRB": "image"}[dataset_name]

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""

        batch[key] = [pytorch_transforms(img) for img in batch[key]]
        # logging.info("""bath key: {}""".format(batch[key]))
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader

def train(model, trainloader, valloader, epochs, learning_rate, device, client_id, t, dataset_name, n_classes):
    """Train the model on the training set."""
    model.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model.train()
    key = {"CIFAR10": "img", "MNIST": "image", "EMNIST": "image", "GTSRB": "image"}[dataset_name]
    for _ in range(epochs):
        loss_total = 0
        correct = 0
        y_true = []
        y_prob = []
        for batch in trainloader:
            # logging.info("""dentro {} labels {}""".format(images, labels))
            images = batch[key]
            labels = batch["label"]
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            # logging.info("""saida: {} true: {}""".format(outputs, labels))
            loss = criterion(outputs, labels)
            loss.backward()
            loss_total += loss.item() * labels.shape[0]
            y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(n_classes)))
            y_prob.append(outputs.detach().cpu().numpy())
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            optimizer.step()
    accuracy = correct / len(trainloader.dataset)
    loss = loss_total / len(trainloader.dataset)
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_prob = y_prob.argmax(axis=1)
    y_true = y_true.argmax(axis=1)
    balanced_accuracy = float(metrics.balanced_accuracy_score(y_true, y_prob))

    train_metrics = {"Train accuracy": accuracy, "Train balanced accuracy": balanced_accuracy, "Train loss": loss, "Train round (t)": t}
    logging.info(train_metrics)

    val_loss, test_metrics = test(model, valloader, device, client_id, t, dataset_name, n_classes)
    results = {
        "val_loss": val_loss,
        "val_accuracy": test_metrics["Accuracy"],
        "val_balanced_accuracy": test_metrics["Balanced accuracy"],
        "train_loss": train_metrics["Train loss"],
        "train_accuracy": train_metrics["Train accuracy"],
        "train_balanced_accuracy": train_metrics["Train balanced accuracy"]
    }
    return results

def train_fedkd(model, trainloader, valloader, epochs, learning_rate, device, client_id, t, dataset_name, n_classes):
    """Train the model on the training set."""
    model.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model.train()
    feature_dim = 512
    W_h = torch.nn.Linear(feature_dim, feature_dim, bias=False)
    MSE = torch.nn.MSELoss()
    key = {"CIFAR10": "img", "MNIST": "image", "EMNIST": "image", "GTSRB": "image"}[dataset_name]
    for _ in range(epochs):
        loss_total = 0
        correct = 0
        y_true = []
        y_prob = []
        for batch in trainloader:
            # logging.info("""dentro {} labels {}""".format(images, labels))
            images = batch[key]
            labels = batch["label"]
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output_student, rep_g, output_teacher, rep = model(images)
            outputs_S1 = F.log_softmax(output_student, dim=1)
            outputs_S2 = F.log_softmax(output_teacher, dim=1)
            outputs_T1 = F.softmax(output_student, dim=1)
            outputs_T2 = F.softmax(output_teacher, dim=1)

            loss_student = criterion(output_student, labels)
            loss_teacher = criterion(output_teacher, labels)
            loss = torch.nn.KLDivLoss()(outputs_S1, outputs_T2) / (loss_student + loss_teacher)
            loss += torch.nn.KLDivLoss()(outputs_S2, outputs_T1) / (loss_student + loss_teacher)
            L_h = MSE(rep, W_h(rep_g)) / (loss_student + loss_teacher)
            loss += loss_student + loss_teacher + L_h
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            loss_total += loss.item() * labels.shape[0]
            y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(n_classes)))
            y_prob.append(outputs.detach().cpu().numpy())
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            optimizer.step()
    accuracy = correct / len(trainloader.dataset)
    loss = loss_total / len(trainloader.dataset)
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_prob = y_prob.argmax(axis=1)
    y_true = y_true.argmax(axis=1)
    balanced_accuracy = float(metrics.balanced_accuracy_score(y_true, y_prob))

    train_metrics = {"Train accuracy": accuracy, "Train balanced accuracy": balanced_accuracy, "Train loss": loss, "Train round (t)": t}
    logging.info(train_metrics)

    val_loss, test_metrics = test(model, valloader, device, client_id, t, dataset_name, n_classes)
    results = {
        "val_loss": val_loss,
        "val_accuracy": test_metrics["Accuracy"],
        "val_balanced_accuracy": test_metrics["Balanced accuracy"],
        "train_loss": train_metrics["Train loss"],
        "train_accuracy": train_metrics["Train accuracy"],
        "train_balanced_accuracy": train_metrics["Train balanced accuracy"]
    }
    return results


def test(model, testloader, device, client_id, t, dataset_name, n_classes):
    """Validate the model on the test set."""
    g = torch.Generator()
    g.manual_seed(t)
    torch.manual_seed(t)
    model.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    correct, loss = 0, 0.0
    y_prob = []
    y_true = []
    key = {"CIFAR10": "img", "MNIST": "image", "EMNIST": "image", "GTSRB": "image"}[dataset_name]
    with torch.no_grad():
        for batch in testloader:
            images = batch[key]
            labels = batch["label"]
            images = images.to(device)
            labels = labels.to(device)
            y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(n_classes)))
            outputs = model(images)
            y_prob.append(outputs.detach().cpu().numpy())
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader.dataset)
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

    y_prob = y_prob.argmax(axis=1)
    y_true = y_true.argmax(axis=1)
    balanced_accuracy = float(metrics.balanced_accuracy_score(y_true, y_prob))

    test_metrics = {"Accuracy": accuracy, "Balanced accuracy": balanced_accuracy, "Loss": loss, "Round (t)": t}
    # logger.info("""metricas cliente {} valores {}""".format(client_id, test_metrics))
    return loss, test_metrics

def test_fedkd(model, testloader, device, client_id, t, dataset_name, n_classes):
        try:
            model.eval()
            criterion = torch.nn.CrossEntropyLoss().to(device)

            correct = 0
            loss = 0
            y_prob = []
            y_true = []

            predictions = np.array([])
            labels = np.array([])

            key = {"CIFAR10": "img", "MNIST": "image", "EMNIST": "image", "GTSRB": "image"}[dataset_name]
            with torch.no_grad():
                for batch in testloader:
                    images = batch[key]
                    labels = batch["label"]
                    images = images.to(device)
                    labels = labels.to(device)
                    y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(n_classes)))
                    output, proto_student, output_teacher, proto_teacher = model(images)
                    if model.new_client:
                        output_teacher = output
                    y_prob.append(output_teacher.detach().cpu().numpy())
                    loss += criterion(output_teacher, labels).item()
                    correct += (torch.max(output_teacher.data, 1)[1] == labels).sum().item()
                    prediction_teacher = torch.argmax(output_teacher, dim=1)
                    predictions = np.append(predictions, prediction_teacher)

            accuracy = correct / len(testloader.dataset)
            loss = loss / len(testloader.dataset)
            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

            y_prob = y_prob.argmax(axis=1)
            y_true = y_true.argmax(axis=1)
            balanced_accuracy = float(metrics.balanced_accuracy_score(y_true, y_prob))

            test_metrics = {"Accuracy": accuracy, "Balanced accuracy": balanced_accuracy, "Loss": loss, "Round (t)": t}
            # logger.info("""metricas cliente {} valores {}""".format(client_id, test_metrics))
            return loss, test_metrics
        except Exception as e:
            print("test_fedkd")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
