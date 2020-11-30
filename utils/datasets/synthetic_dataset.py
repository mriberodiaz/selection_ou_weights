import collections
import numpy as np

import tensorflow as tf
import tensorflow_federated as tff
import math
import os
import sys
import random
from tqdm import trange




def get_logistic_regression_dataset(dimension, num_samples_per_client, num_clients):
    """Creates logistic regression datset.

    Returns:
    A `(train, test)` tuple where `train` and `test` are `tf.data.Dataset` representing 
    the test data of all clients.
    """
    beta = tf.random.normal(shape = (dimension,))
    # include test clients
    total_clients = num_clients//10 + num_clients

    X=tf.random.normal(shape=(
    num_samples_per_client*total_clients,dimension))
    y = 1/(1+np.exp(-tf.linalg.matvec(X,beta)))
    y_labels = tf.cast(tf.round(y),tf.int32)

    return X,y_labels, beta




def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex


def generate_synthetic(alpha, beta, iid):

    dimension = 60
    NUM_CLASS = 10
    NUM_USER = 100
    
    samples_per_user = np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 50
    #print(samples_per_user)
    num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]


    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    B = np.random.normal(0, beta, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)
        #print(mean_x[i])

    if iid == 1:
        W_global = np.random.normal(0, 1, (dimension, NUM_CLASS))
        b_global = np.random.normal(0, 1,  NUM_CLASS)

    for i in range(NUM_USER):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1,  NUM_CLASS)

        if iid == 1:
            W = W_global
            b = b_global

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

        #print("{}-th users has {} exampls".format(i, len(y_split[i])))


    return X_split, y_split


def generate_federated_softmax_data(batch_size,
    client_epochs_per_round, 
    test_batch_size):
    NUM_USER = 100
    num_test_clients=10

    X, y = generate_synthetic(alpha=1, beta=1, iid=0)     # synthetic (1,1)
    #X, y = generate_synthetic(alpha=0, beta=0, iid=1)      # synthetic_IID

    def get_client_data(client_id):
        return tf.data.Dataset.from_tensor_slices(
            collections.OrderedDict(x= X[client_id],
                            y= y[client_id],
                           ))

    clients_ids = np.arange(NUM_USER).tolist()
    federated_data = tff.simulation.ClientData.from_clients_and_fn(clients_ids, get_client_data)
    train, test = federated_data.train_test_client_split(federated_data, num_test_clients)

    def preprocess_train_dataset(dataset):
        return dataset.shuffle(buffer_size=418).repeat(
            count=client_epochs_per_round).batch(
                batch_size, drop_remainder=False)

    def preprocess_test_dataset(dataset):
        return dataset.batch(test_batch_size, drop_remainder=False)

    train = train.preprocess(preprocess_train_dataset)
    test = preprocess_test_dataset( test.create_tf_dataset_from_all_clients())

    return train, test


def create_lr_federatedClientData(dimension, 
    num_samples_per_client, 
    num_clients, 
    batch_size,
    client_epochs_per_round, 
    test_batch_size):
    num_test_clients = num_clients//10
    X, y , beta = get_logistic_regression_dataset(dimension, num_samples_per_client, num_clients+num_test_clients)
    def get_client_data(client_id):
        return tf.data.Dataset.from_tensor_slices(
            collections.OrderedDict(x= X[client_id*num_samples_per_client:(client_id+1)*num_samples_per_client],
                            y= y[client_id*num_samples_per_client:(client_id+1)*num_samples_per_client],
                           ))
    clients_ids = np.arange(num_clients).tolist()
    federated_data = tff.simulation.ClientData.from_clients_and_fn(clients_ids, get_client_data)
    train, test = federated_data.train_test_client_split(federated_data, num_test_clients)

    def element_fn(element):
        return collections.OrderedDict(
            x=element['x'], y=element['y'][..., tf.newaxis])

    def preprocess_train_dataset(dataset):
        return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
            count=client_epochs_per_round).batch(
                batch_size, drop_remainder=False)

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(test_batch_size, drop_remainder=False)

    train = train.preprocess(preprocess_train_dataset)
    test = preprocess_test_dataset( test.create_tf_dataset_from_all_clients())

    return train, test, beta