# QuAFL: Quantized Asynchronous Federated Learning
### This repository contains the code for the paper "QuAFL: Federated Averaging Can Be Both Asynchronous and Communication-Efficient".

### It is a time-based simulator for running different federated learning algorithms. 

Three FL algorithms compared in the paper, QuAFL, FedAvg, and FedBuff, are implemented in trainer.py. The code tracks the performance of 
each algorithm by evaluating measuring the *loss* and *accuracy* of server's model with respect to *simulation time*, *server steps*, *total number of local steps*, and *total bits transmitted*.

To run the code, you should identify the setups in the Quantized_fl.ipynb. Note that the code is designed to use the same train/validation split, initial learning rate, and simulation time for all the configurations.
All the hyper-parameters described in section 5.1 of the paper, as well as *time limit (simulation time)*, *initial learning rate*, and the *GPU ids* on which the models should be stored, can be for each setup. 
A sample configuration for each algorithm is demonstrated in Quantized_FL.ipynb as an example. 

Description of other files:

* server: Server's aggregation, sending, and receiving methods, supporting both model and update transmission

* client: Client's local step, updating, sending model/update methods

* quantizer: encode and decode functions for *lattice* and [*QSGD*](https://arxiv.org/abs/1610.02132) quantizers

### Models and Datasets

For the experiments, we run the algorithms on four datasets: MNIST, FMNIST, CIFAR-10, and CelebA. The dataset and model preparations are done in dataset_manager.py and model_manager.py, respectively. 
For MNIST, FMNIST, and CIFAR-10 experiments, we distribute the data uniformly at random among the clients.

For the large-scale runs, we used CelebA dataset as described in [LEAF](https://github.com/TalwalkarLab/leaf). We prepared a list of datasets separated based on the celebrity-id, such that each of them contains at least 10 images.
For each run on CelebA, we distribute the training part of these datasets among the clients and gather all the validation parts for the server's model evaluation. Consequently, the training sets of the clients are disjoint.


