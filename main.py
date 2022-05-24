import argparse
import logging
from server import Server
from utils import config

# Set up parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str, default='./configs/MNIST/mnist.json',
    #                     help='Federated learning configuration file.')
    parser.add_argument('-c', '--config', type=str, default='./configs/FashionMNIST/fashionmnist.json',
                        help='Federated learning configuration file.') #for FashionMNIST
    # parser.add_argument('-c', '--config', type=str, default='./configs/CIFAR-10/cifar-10.json',
    #                     help='Federated learning configuration file.') #for CIFAR-10
    parser.add_argument('-l', '--log', type=str, default='INFO',
                        help='Log messages level.')
    # parser.add_argument('-d', '--dataset', type=str, default='MNIST',
                       # help='the name of dataset')
    parser.add_argument('-d', '--dataset', type=str, default='FashionMNIST',
                        help='the name of dataset') #for FashionMMIST
    # parser.add_argument('-d', '--dataset', type=str, default='CIFAR-10',
    #                     help='the name of dataset') #for CIFAR-10

    args = parser.parse_args()
    # Set logging
    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()),
        datefmt='%H:%M:%S')
    logging.info("config:{},  log:{}".format(args.config, args.log))
    # load config
    # print("main args.config : ", args.config) #check
    config = config.Config(args.config)
    server = Server(config)
    server.run()
