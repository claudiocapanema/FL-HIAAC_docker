import json
from collections import OrderedDict

import torch
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomResizedCrop, RandomAffine, ColorJitter,  Normalize, ToTensor, RandomRotation, Lambda
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import numpy as np
import sys
from utils.models import CNN, CNN_3, CNNDistillation, GRU, LSTM


import logging


logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

DATASET_INPUT_MAP = {"CIFAR10": "img", "MNIST": "image", "EMNIST": "image", "GTSRB": "image", "Gowalla": "sequence", "WISDM-W": "sequence", "ImageNet": "image"}

def load_model(model_name, dataset, strategy, device):
    try:
        num_classes = {'EMNIST': 47, 'MNIST': 10, 'CIFAR10': 10, 'GTSRB': 43, 'WISDM-W': 12, 'WISDM-P': 12, 'Tiny-ImageNet': 200,
         'ImageNet100': 15, 'ImageNet': 15, "ImageNet_v2": 15, "Gowalla": 7}[dataset]
        if model_name == 'CNN':
            if dataset in ['MNIST']:
                input_shape = 1
                mid_dim = 256*4
                logger.info("""leu mnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset in ['EMNIST']:
                input_shape = 1
                mid_dim = 256*4
                logger.info("""leu emnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset in ['GTSRB']:
                input_shape = 3
                mid_dim = 36*4
                logger.info("""leu gtsrb com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset in ["ImageNet"]:
                input_shape=3
                mid_dim=1600
            elif dataset == "CIFAR10":
                input_shape = 3
                mid_dim = 400*4
                logger.info("""leu cifar com {} {} {}""".format(input_shape, mid_dim, num_classes))
            return CNN(input_shape=input_shape, num_classes=num_classes, mid_dim=mid_dim)
        elif model_name == 'CNN_3':
            if dataset in ['MNIST']:
                input_shape = 1
                mid_dim = 4
                logger.info("""leu mnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset in ['EMNIST']:
                input_shape = 1
                mid_dim = 4
                logger.info("""leu emnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset in ['GTSRB']:
                input_shape = 3
                mid_dim = 16
                logger.info("""leu gtsrb com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset == "ImageNet":
                input_shape = 3
                mid_dim = 16
                logger.info("""leu imagenet com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset == "CIFAR10":
                input_shape = 3
                mid_dim = 16
                logger.info("""leu cifar com {} {} {}""".format(input_shape, mid_dim, num_classes))

            if "FedKD" in strategy:
                return CNNDistillation(input_shape=input_shape, mid_dim=mid_dim, num_classes=num_classes, dataset=dataset)
            else:
                return CNN_3(input_shape=input_shape, num_classes=num_classes, mid_dim=mid_dim)

        elif model_name == "gru":
            if dataset in ["WISDM-W", "WISDM-P"]:
                return GRU(6, num_layers=1, hidden_size=2, sequence_length=200, num_classes=num_classes)

        elif model_name == "lstm":
            if dataset in ["Gowalla"]:
                return LSTM(4, device=device, num_layers=1, hidden_size=1, sequence_length=10, num_classes=num_classes)

        raise ValueError("""Model not found for model {} and dataset {}""".format(model_name, dataset))

    except Exception as e:
        logger.info("""load_model error""")
        logger.info("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


fds = None

def get_weights(net):
    try:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    except Exception as e:
        logger.info("get_weights error")
        logger.info("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def get_weights_fedkd(net):
    try:
        return [val.cpu().numpy() for _, val in net.student.state_dict().items()]
    except Exception as e:
        logger.info("get_weights_fedkd error")
        logger.info("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def set_weights(net, parameters):
    try:
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
    except Exception as e:
        logger.info("set_weights error")
        logger.info("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def set_weights_fedkd(net, parameters):
    try:
        params_dict = zip(net.student.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.student.load_state_dict(state_dict, strict=True)
    except Exception as e:
        logger.info("set_weights_fedkd error")
        logger.info("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def load_data(dataset_name: str, alpha: float, partition_id: int, num_partitions: int, batch_size: int,
              data_sampling_percentage: int):
    try:
        """Load partition CIFAR10 data."""
        # Only initialize `FederatedDataset` once
        logger.info(
            """Loading {} {} {} {} {} {} data.""".format(dataset_name, partition_id, num_partitions, batch_size, data_sampling_percentage, alpha))
        # global fds
        # if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label",

                                           alpha=alpha, min_partition_size=10,

                                           self_balancing=True)
        fds = FederatedDataset(
            dataset={"EMNIST": "claudiogsc/emnist_balanced", "CIFAR10": "uoft-cs/cifar10", "MNIST": "ylecun/mnist",
                     "GTSRB": "claudiogsc/GTSRB", "Gowalla": "claudiogsc/Gowalla-State-of-Texas",
                     "WISDM-W": "claudiogsc/WISDM-W", "ImageNet": "claudiogsc/ImageNet-15_household_objects"}[dataset_name],
            partitioners={"train": partitioner},
        )
        partition = fds.load_partition(partition_id)
        # Divide data on each node: 80% train, 20% test
        test_size = 1 - data_sampling_percentage
        partition_train_test = partition.train_test_split(test_size=test_size, seed=42)

        if dataset_name in ["CIFAR10", "MNIST", "EMNIST", "GTSRB", "ImageNet", "WISDM-W", "Gowalla"]:
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
                        ),
                "ImageNet": Compose(
                        [

                            Resize(32),
                            RandomHorizontalFlip(),
                            ToTensor(),
                            Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                            # transforms.Resize((32, 32)),
                            # transforms.ToTensor(),
                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]
                    ),
                "WISDM-W": Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.float32))),
                "Gowalla": Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.float32))),

            }[dataset_name]

        # import torchvision.datasets as datasets
        # datasets.EMNIST
        key = DATASET_INPUT_MAP[dataset_name]

        def apply_transforms(batch):
            """Apply transforms to the partition from FederatedDataset."""

            batch[key] = [pytorch_transforms(img) for img in batch[key]]
            # logger.info("""bath key: {}""".format(batch[key]))
            return batch

        if dataset_name in ["CIFAR10", "MNIST", "EMNIST", "GTSRB", "ImageNet", "WISDM-W", "Gowalla"]:
            partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
        return trainloader, testloader

    except Exception as e:
        logger.info("load_data error")
        logger.info("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def train(model, trainloader, valloader, optimizer, epochs, learning_rate, device, client_id, t, dataset_name, n_classes):
    try:
        """Train the utils on the training set."""
        model.to(device)  # move utils to GPU if available
        criterion = torch.nn.CrossEntropyLoss().to(device)
        model.train()
        key = DATASET_INPUT_MAP[dataset_name]
        for _ in range(epochs):
            loss_total = 0
            correct = 0
            y_true = []
            y_prob = []
            for batch in trainloader:
                # logger.info("""dentro {} labels {}""".format(images, labels))
                x = batch[key]
                labels = batch["label"]
                # logger.info("""tamanho images {} tamanho labels {}""".format(images.shape, labels.shape))
                x = x.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(x)
                # logger.info("""saida: {} true: {}""".format(outputs, labels))
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
        logger.info(train_metrics)

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

    except Exception as e:
        logger.info("train error")
        logger.info("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def train_fedkd(model, trainloader, valloader, epochs, learning_rate, device, client_id, t, dataset_name, n_classes):
    """Train the utils on the training set."""
    try:
        model.to(device)  # move utils to GPU if available
        # utils.teacher.to(device)
        # utils.student.to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        model.train()
        feature_dim = 512
        W_h = torch.nn.Linear(feature_dim, feature_dim, bias=False).to(device)
        MSE = torch.nn.MSELoss().to(device)
        key = DATASET_INPUT_MAP[dataset_name]
        logger.info("""Inicio train_fedkd client {}""".format(client_id))
        for _ in range(epochs):
            loss_total = 0
            correct = 0
            y_true = []
            y_prob = []
            for batch in trainloader:
                # logger.info("""dentro {} labels {}""".format(images, labels))
                x = batch[key]
                labels = batch["label"]
                x = x.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                output_student, rep_g, output_teacher, rep = model(x)
                outputs_S1 = F.log_softmax(output_student, dim=1)
                outputs_S2 = F.log_softmax(output_teacher, dim=1)
                outputs_T1 = F.softmax(output_student, dim=1)
                outputs_T2 = F.softmax(output_teacher, dim=1)

                loss_student = criterion(output_student, labels)
                loss_teacher = criterion(output_teacher, labels)
                # loss_1 = torch.nn.KLDivLoss()(outputs_S1, outputs_T2) / (loss_student + loss_teacher)
                # loss_2 = torch.nn.KLDivLoss()(outputs_S2, outputs_T1) / (loss_student + loss_teacher)
                # L_h = MSE(rep, W_h(rep_g)) / (loss_student + loss_teacher)
                loss = loss_student + loss_teacher
                # loss = loss_teacher + loss_student + L_h + loss_1 + loss_2
                loss.backward()
                loss_total += loss.item() * labels.shape[0]
                y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(n_classes)))
                y_prob.append(output_teacher.detach().cpu().numpy())
                correct += (torch.max(output_teacher.data, 1)[1] == labels).sum().item()
                optimizer.step()
        accuracy = correct / len(trainloader.dataset)
        loss = loss_total / len(trainloader.dataset)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        y_prob = y_prob.argmax(axis=1)
        y_true = y_true.argmax(axis=1)
        balanced_accuracy = float(metrics.balanced_accuracy_score(y_true, y_prob))

        train_metrics = {"Train accuracy": accuracy, "Train balanced accuracy": balanced_accuracy, "Train loss": loss, "Train round (t)": t}
        logger.info(train_metrics)

        val_loss, test_metrics = test_fedkd(model, valloader, device, client_id, t, dataset_name, n_classes)
        results = {
            "val_loss": val_loss,
            "val_accuracy": test_metrics["Accuracy"],
            "val_balanced_accuracy": test_metrics["Balanced accuracy"],
            "train_loss": train_metrics["Train loss"],
            "train_accuracy": train_metrics["Train accuracy"],
            "train_balanced_accuracy": train_metrics["Train balanced accuracy"]
        }
        return results

    except Exception as e:
        logger.info("""Error on train_fedkd""")
        logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def test(model, testloader, device, client_id, t, dataset_name, n_classes):
    try:
        """Validate the utils on the test set."""
        g = torch.Generator()
        g.manual_seed(t)
        torch.manual_seed(t)
        model.eval()
        model.to(device)  # move utils to GPU if available
        criterion = torch.nn.CrossEntropyLoss().to(device)
        correct, loss = 0, 0.0
        y_prob = []
        y_true = []
        key = DATASET_INPUT_MAP[dataset_name]
        with torch.no_grad():
            for batch in testloader:
                x = batch[key]
                labels = batch["label"]
                x = x.to(device)
                labels = labels.to(device)
                y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(n_classes)))
                outputs = model(x)
                y_prob.append(outputs.detach().cpu().numpy())
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader.dataset)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        y_prob = y_prob.argmax(axis=1)
        y_true = y_true.argmax(axis=1)
        balanced_accuracy = float(metrics.balanced_accuracy_score(y_true, y_prob))

        test_metrics = {"Accuracy": accuracy, "Balanced accuracy": balanced_accuracy, "Loss": loss, "Round (t)": t}
        # logger.info("""metricas cliente {} valores {}""".format(client_id, test_metrics))
        return loss, test_metrics

    except Exception as e:
        logger.info(" error")
        logger.info("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def test_fedkd(model, testloader, device, client_id, t, dataset_name, n_classes):
        try:
            model.to(device)  # move utils to GPU if available
            # utils.teacher.to(device)
            # utils.student.to(device)
            model.eval()
            criterion = torch.nn.CrossEntropyLoss().to(device)

            correct = 0
            loss = 0
            y_prob = []
            y_true = []

            key = DATASET_INPUT_MAP[dataset_name]
            with torch.no_grad():
                for batch in testloader:
                    x = batch[key]
                    labels = batch["label"]
                    x = x.to(device)
                    labels = labels.to(device)
                    y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(n_classes)))
                    output, proto_student, output_teacher, proto_teacher = model(x)
                    y_prob.append(output_teacher.detach().cpu().numpy())
                    loss += criterion(output_teacher, labels).item()
                    correct += (torch.sum(torch.argmax(output_teacher, dim=1) == labels)).item()

            accuracy = correct / len(testloader.dataset)
            loss = loss / len(testloader.dataset)
            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            # test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

            y_prob = y_prob.argmax(axis=1)
            y_true = y_true.argmax(axis=1)
            balanced_accuracy = float(metrics.balanced_accuracy_score(y_true, y_prob))

            test_metrics = {"Accuracy": accuracy, "Balanced accuracy": balanced_accuracy, "Loss": loss, "Round (t)": t}
            # logger.info("""metricas cliente {} valores {}""".format(client_id, test_metrics))
            return loss, test_metrics
        except Exception as e:
            logger.info("Error test_fedkd")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def test_fedkd_fedpredict(lt, model, testloader, device, client_id, t, dataset_name, n_classes):
    try:
        model.to(device)  # move utils to GPU if available
        # utils.teacher.to(device)
        # utils.student.to(device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss().to(device)

        correct = 0
        loss = 0
        y_prob = []
        y_true = []

        key = DATASET_INPUT_MAP[dataset_name]
        with torch.no_grad():
            for batch in testloader:
                x = batch[key]
                labels = batch["label"]
                x = x.to(device)
                labels = labels.to(device)
                y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(n_classes)))
                output, proto_student, output_teacher, proto_teacher = model(x)
                if lt == 0:
                    output_teacher = output
                y_prob.append(output_teacher.detach().cpu().numpy())
                loss += criterion(output_teacher, labels).item()
                correct += (torch.sum(torch.argmax(output_teacher, dim=1) == labels)).item()

        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader.dataset)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        y_prob = y_prob.argmax(axis=1)
        y_true = y_true.argmax(axis=1)
        balanced_accuracy = float(metrics.balanced_accuracy_score(y_true, y_prob))

        test_metrics = {"Accuracy": accuracy, "Balanced accuracy": balanced_accuracy, "Loss": loss, "Round (t)": t}
        # logger.info("""metricas cliente {} valores {}""".format(client_id, test_metrics))
        return loss, test_metrics
    except Exception as e:
        logger.info("Error test_fedkd_fedpredict")
        logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
