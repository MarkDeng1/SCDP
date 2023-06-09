#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--exp_name', type=str, default='exp',
                        help="the name of the current experiment")
    parser.add_argument('--epochs', type=int, default=95,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--norm_mean', type=float, default=0.5,
                        help="normalize the data to norm_mean")
    parser.add_argument('--norm_std', type=float, default=0.5,
                        help="normalize the data to norm_std")     
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help="testset batch size") 

    # model arguments
    parser.add_argument('--model', type=str, default='simpleCNN', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    parser.add_argument('--dim_hidden', type=int, default=64,
                        help='dim_hidden')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset")
    parser.add_argument('--data', type=str, default='cifar', help="name \
                        of dataset") 
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=29,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--class_per_user', type=int, default=1, help="class_per_user")
    parser.add_argument('--mode', type=str, default='SIGNSGD', help="SIGNSGD|DP-SIGNSGD|EF-DP-SIGNSGD|FedAvg")
    parser.add_argument('--eps', type=float, default=1,
                        help='DP eps')
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='DP delta')
    parser.add_argument('--l2_norm_clip', type=float, default=1,
                        help='l2_norm_clip')
    parser.add_argument('--Byzantine', type=float, default=0,
                        help='Byzantine')
    parser.add_argument('--gamma', type=float, default=1,
                        help='gamma')
    parser.add_argument('--error_decay', type=float, default=1,
                        help='error_decay')
    parser.add_argument('--weighted', type=int, default=0, help='weighted')
    parser.add_argument('--server_momentum', type=int, default=0, help='[0,1]')
    args = parser.parse_args()
    return args
