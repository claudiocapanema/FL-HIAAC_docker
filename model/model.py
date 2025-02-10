import tensorflow as tf
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from tensorflow.python.ops.metrics_impl import accuracy
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import numpy as np


import logging


logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Class for the model. In this case, we are using the MobileNetV2 model from Keras
class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

fds = None

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def load_data(partition_id: int, num_partitions: int, batch_size: int, data_sampling_percentage: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    test_size = 1 - data_sampling_percentage
    partition_train_test = partition.train_test_split(test_size=test_size, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    g = torch.Generator()
    g.manual_seed(partition_id)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True, g=g
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader

def train(net, trainloader, valloader, epochs, learning_rate, device, client_id, t):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    for _ in range(epochs):
        loss_total = 0
        correct = 0
        y_true = []
        y_prob = []
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            loss_total += loss.item() * labels.shape[0]
            y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(10)))
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

    val_loss, test_metrics = test(net, valloader, device, client_id, t)

    results = {
        "val_loss": val_loss,
        "val_accuracy": test_metrics["Accuracy"],
        "val_balanced_accuracy": test_metrics["Balanced accuracy"],
        "train_loss": train_metrics["Train loss"],
        "train_accuracy": train_metrics["Train accuracy"],
        "train_balanced_accuracy": train_metrics["Train balanced accuracy"]
    }
    return results


def test(net, testloader, device, client_id, t):
    """Validate the model on the test set."""
    g = torch.Generator()
    g.manual_seed(t)
    torch.manual_seed(t)
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    y_prob = []
    y_true = []
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(10)))
            outputs = net(images)
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
