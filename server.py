import torch
import logging
from models import models
from utils.config import Config
from clients import Client
import copy
import numpy as np

import matplotlib.pyplot as plt

class Server:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model()
        print("model : ", self.model)
        self.clients = None
        self.client_index = []
        self.target_round = -1

        self.array_train_acc = []
        self.array_test_acc = []
        self.array_test_loss = []

    def run(self):
        self.connect_clients()
        # communication rounds
        for round in (range(1, self.config.fl.rounds + 1)):
            logging.info("-" * 22 + "round {}".format(round) + "-" * 30)
            # select clients which participate training
            selected = self.clients_selection()
            logging.info("selected clients:{}".format(selected))
            info = self.clients.train(selected)

            logging.info("aggregate weights")
            # update glob model
            glob_weights = self.fed_avg(info)
            self.model.load_state_dict(glob_weights)
            train_acc = self.getacc(info)
            test_acc, test_loss = self.test()

            self.array_train_acc.append(train_acc)
            self.array_test_acc.append(test_acc)
            self.array_test_loss.append(test_loss)

            logging.info(
                "training acc: {:.4f},test acc: {:.4f}, test_loss: {:.4f}\n".format(train_acc, test_acc, test_loss))
            if test_acc > self.config.fl.target_accuracy:
                self.target_round = round
                logging.info("target achieved")
                break

            # broadcast glob weights
            self.clients.update(glob_weights)
        

        self.show_graph()
    
    def fed_avg(self, info):
        weights = info["weights"]
        length = info["len"]
        w_avg = copy.deepcopy(weights[0])
        for k in w_avg.keys():
            w_avg[k] *= length[0]
            for i in range(1, len(weights)):
                w_avg[k] += weights[i][k] * length[i]
            w_avg[k] = w_avg[k] / (sum(length))
        return w_avg

    def clients_selection(self):
        # randomly selection
        frac = self.config.clients.fraction
        n_clients = max(1, int(self.config.clients.total * frac))
        training_clients = np.random.choice(self.client_index, n_clients, replace=False)
        return training_clients

    def load_model(self):
        model_path = self.config.paths.model
        dataset = self.config.dataset
        logging.info('dataset: {}'.format(dataset))

        # Set up global model
        model = models.get_model(dataset)
        logging.debug(model)
        return model

    def connect_clients(self):
        self.clients = Client(self.config)
        self.client_index = self.clients.clients_to_server()
        self.clients.get_model(self.model)
        self.clients.load_data()

    def test(self):
        return self.clients.test()

    def getacc(self, info):
        corrects = sum(info["corrects"])
        total_samples = sum(info["len"])
        return corrects / total_samples

    def show_graph(self):
        # print("array_train_acc : ", self.array_train_acc)
        # print("array_test_acc : ", self.array_test_acc)
        # print("array_test_loss : ", self.array_test_loss)

        fig, acc_graph = plt.subplots()

        acc_graph.plot(self.array_train_acc, 'g', label='train_accuracy')
        acc_graph.plot(self.array_test_acc, 'r', label='test_accuracy')
        acc_graph.set_xlabel('round (1 round = 5 epochs)')
        acc_graph.set_ylabel('accuracy')
        acc_graph.legend(loc='lower left')

        # loss_graph.plot(self.array_test_loss, 'r', label='test_loss')
        # loss_graph.set_xlabel('round (1 round = 5 epochs)')
        # loss_graph.set_ylabel('loss')
        # loss_graph.legend(loc='lower left')

        plt.show()


if __name__ == "__main__":
    # config = Config("configs/MNIST/mnist.json") #for MNIST
    config = Config("configs/FashionMNIST/fashionmnist.json") #for FashionMNIST
    # config = Config("configs/CIFAR-10/cifar-10.json") #for CIFAR-10
    server = Server(config)
    server.run()
