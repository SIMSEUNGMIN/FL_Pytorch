from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Subset
from torch._utils import _accumulate
import random
import numpy as np


class MNIST:
    def __init__(self, config):
        self.config = config
        print(self.config.paths.data)
        self.path = str(self.config.paths.data) + '/' + self.config.dataset
        self.trainset = None
        self.testset = None

    def load_data(self, IID=True):
        # # CIFAR10
        # self.trainset = datasets.CIFAR10(
        #     self.path, train=True, download=True, transform=transforms.Compose([
        #       transforms.Resize((227, 227)),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0,5))
        #     ])
        # )
        # self.testset = datasets.CIFAR10(
        #     self.path, train=False, download=True, transform=transforms.Compose([
        #       transforms.Resize((227, 227)),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0,5))
        #     ])
        # )
        # total_clients = self.config.clients.total
        # total_sample = self.trainset.data.shape[0]
        # # number of samples on each client
        # length = [total_sample // total_clients] * total_clients
        # if IID:
        #     spilted_train = random_split(self.trainset, length)

        # else:
        #     print("None-IID")
        #     if sum(length) != len(self.trainset):
        #         raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
        #     index = []
        #     for i in range(10):
        #         index.append([])

        #     i = 0
        #     for img, label in self.trainset:
        #         index[label].append(i)
        #         i += 1

        #     indices = np.array([elem for c_list in index for elem in c_list]).reshape(-1, 200)

        #     np.random.shuffle(indices)
        #     indices = indices.flatten()
        #     print(indices.shape)

        #     spilted_train = [Subset(self.trainset, indices[offset - length:offset]) for offset, length in
        #                      zip(_accumulate(length), length)]
        #     print(len(spilted_train))
        # return spilted_train, self.testset

        # MNIST
        # self.trainset = datasets.MNIST(
        #     self.path, train=True, download=True, transform=transforms.Compose([
        #         transforms.RandomRotation(15),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,))
        #     ]))
        # self.testset = datasets.MNIST(
        #     self.path, train=False, transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,))
        #     ]))
        # total_clients = self.config.clients.total
        # total_sample = self.trainset.data.shape[0]
        # # number of samples on each client
        # length = [total_sample // total_clients] * total_clients
        # if IID:
        #     spilted_train = random_split(self.trainset, length)

        # else:
        #     print("None-IID")
        #     if sum(length) != len(self.trainset):
        #         raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
        #     index = []
        #     for i in range(10):
        #         index.append([])

        #     i = 0
        #     for img, label in self.trainset:
        #         index[label].append(i)
        #         i += 1

        #     indices = np.array([elem for c_list in index for elem in c_list]).reshape(-1, 200)

        #     np.random.shuffle(indices)
        #     indices = indices.flatten()
        #     print(indices.shape)

        #     spilted_train = [Subset(self.trainset, indices[offset - length:offset]) for offset, length in
        #                      zip(_accumulate(length), length)]
        #     print(len(spilted_train))
        # return spilted_train, self.testset

        # FashionMNIST
        self.trainset = datasets.FashionMNIST(
            self.path, train=True, download=True, transform=transforms.Compose([
                transforms.Resize((227, 227)),
                # transforms.RandomRotation(15),
                transforms.ToTensor(), # 데이터를 0에서 255까지 있는 값을 0에서 1사이로 변환
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        self.testset = datasets.FashionMNIST(
            self.path, train=False, transform=transforms.Compose([
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        total_clients = self.config.clients.total
        total_sample = self.trainset.data.shape[0]
        # number of samples on each client
        length = [total_sample // total_clients] * total_clients
        if IID:
            spilted_train = random_split(self.trainset, length)

        else:
            print("None-IID")
            if sum(length) != len(self.trainset):
                raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
            index = []
            for i in range(10):
                index.append([])

            i = 0
            for img, label in self.trainset:
                index[label].append(i)
                i += 1

            indices = np.array([elem for c_list in index for elem in c_list]).reshape(-1, 200)

            np.random.shuffle(indices)
            indices = indices.flatten()
            print(indices.shape)

            spilted_train = [Subset(self.trainset, indices[offset - length:offset]) for offset, length in
                             zip(_accumulate(length), length)]
            print(len(spilted_train))
        return spilted_train, self.testset



def get_data(dataset, config):
    if dataset == "MNIST":
        return MNIST(config).load_data(IID=config.data.IID)
    elif dataset == "FashionMNIST":
        # return FashionMNIST(config).load_data(IID=config.data.IID)
        return MNIST(config).load_data(IID=config.data.IID)
    elif dataset == "CIFAR-10":
        # return CIFAR10config).load_data(IID=config.data.IID)
        return MNIST(config).load_data(IID=config.data.IID)
