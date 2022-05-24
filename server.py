import torch
import logging
from models import models
from utils.config import Config
from clients import Client
import copy
import numpy as np

import csv
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
        
        # for data processing
        self.array_train_acc = np.round(self.array_train_acc, 3)
        self.array_test_acc = np.round(self.array_test_acc, 3)
        self.array_test_loss = np.round(self.array_test_loss, 3)

        # for data visualization
        self.write_file()
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

    def write_file(self):
        f = open("C:/Users/user/Desktop/AlexNet/FashionMNIST/IID/IID_1000R_RESULT.csv", "w", newline="")
        writer = csv.writer(f)
        writer.writerow(['array_train_acc'])
        writer.writerow(self.array_train_acc)
        writer.writerow(['array_test_acc'])
        writer.writerow(self.array_test_acc)
        writer.writerow(['array_test_loss'])
        writer.writerow(self.array_test_loss)
        f.close()
        

    def show_graph(self):

        # for test printing
        # print("array_train_acc : ", self.array_train_acc)
        # print("array_test_acc : ", self.array_test_acc)
        # print("array_test_loss : ", self.array_test_loss)

        # setting graph shape
        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches((15,15))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        plt.suptitle('AlexNet/FashionMNIST/IID/lr=0.01', fontsize=15)

        # train accuracy graph
        # axes[0,0].set_title('AlexNet/FashionMNIST/NON-IID/lr=0.01')
        axes[0,0].plot(self.array_train_acc, 'g', label='train_accuracy')
        axes[0,0].set_xlabel('round (1 round = 5 epochs)')
        axes[0,0].set_ylabel('accuracy')
        axes[0,0].legend(loc='lower left')

        # test accuracy graph
        # axes[0,1].set_title('AlexNet/FashionMNIST/NON-IID/lr=0.01')
        axes[0,1].plot(self.array_test_acc, 'r', label='test_accuracy')
        axes[0,1].set_xlabel('round (1 round = 5 epochs)')
        axes[0,1].set_ylabel('accuracy')
        axes[0,1].legend(loc='lower left')

        # train and test accuracy graph
        # axes[1,0].set_title('AlexNet_FashionMNIST_NON-IID_lr=0.01')
        axes[1,0].plot(self.array_train_acc, 'g', label='train_accuracy')
        axes[1,0].plot(self.array_test_acc, 'r', label='test_accuracy')
        axes[1,0].set_xlabel('round (1 round = 5 epochs)')
        axes[1,0].set_ylabel('accuracy')
        axes[1,0].legend(loc='lower left')
        
        # test loss graph
        # axes[1,1].set_title('AlexNet/FashionMNIST/NON-IID/lr=0.01')
        axes[1,1].plot(self.array_test_loss, 'b', label='test_loss')
        axes[1,1].set_xlabel('round (1 round = 5 epochs)')
        axes[1,1].set_ylabel('loss')
        axes[1,1].legend(loc='lower left')

        plt.show()


if __name__ == "__main__":
    # config = Config("configs/MNIST/mnist.json") #for MNIST
    config = Config("configs/FashionMNIST/fashionmnist.json") #for FashionMNIST
    # config = Config("configs/CIFAR-10/cifar-10.json") #for CIFAR-10
    server = Server(config)
    server.run()
