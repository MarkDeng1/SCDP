#coding=utf-8
import torch
from torch.optim import Optimizer
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal,mnist_noniid_class,cifar_noniid_class
from sampling import cifar_iid, cifar_noniid
from update import LocalUpdate, test_inference
from scipy.special import erf
import time
import os
from math import exp, sqrt
import numpy as np
#Signum with majority vote

class signSGD(Optimizer):

    def __init__(self, params, lr=0.01, rand_zero=True):
        defaults = dict(lr=lr)
        self.rand_zero = rand_zero
        super(signSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # take sign of gradient
                grad = torch.sign(p.grad)

                # randomise zero gradients to ±1
                if self.rand_zero:
                    grad[grad==0] = torch.randint_like(grad[grad==0], low=0, high=2)*2 - 1
                    assert not (grad==0).any()
                
                # make update
                p.data -= group['lr'] * grad

        return loss



class signNum(Optimizer):

    def __init__(self, params, args, **kwargs):

        lr = args.lr
        momentum = args.momentum
        weight_decay = args.weight_decay
        compression_buffer = args.compress
        all_reduce = args.all_reduce
        local_rank = args.local_rank
        gpus_per_machine = args.gpus_per_machine

        self.compression_buffer = compression_buffer
        self.all_reduce = all_reduce
        self.signum = args.signum

        self.args = args

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay)

        super(signNum, self).__init__(params, defaults)

        self.MB = 1024 * 1024
        self.bucket_size = 100 * self.MB

        if self.compression_buffer:
            import compressor

            self.compressor = compressor.compressor(using_cuda = True, local_rank = local_rank, cpp_extend_load = args.cpp_extend_load)
            self.local_rank = local_rank
            self.global_rank = dist.get_rank()
            self.local_dst_in_global = self.global_rank - self.local_rank

            self.inter_node_group = []
            self.nodes = dist.get_world_size() // gpus_per_machine

            self.intra_node_group_list = []
            for index in range(self.nodes):
                # set inter_node_group
                self.inter_node_group.append(0 + index * gpus_per_machine)
                # set all intra_node_group
                intra_node_group_temp = []
                for intra_index in range(gpus_per_machine):
                    intra_node_group_temp.append(intra_index + index * gpus_per_machine)
                intra_node_group_temp = dist.new_group(intra_node_group_temp)
                self.intra_node_group_list.append(intra_node_group_temp)

                if self.local_dst_in_global == 0 + index * gpus_per_machine:
                    self.nodes_rank = index


            #self.intra_node_list = self.intra_node_group
            self.inter_node_list = self.inter_node_group
            self.inter_node_group_list = []
            for index in range(len(self.inter_node_list)):
                if index is not 0:
                    temp = dist.new_group([self.inter_node_list[0],self.inter_node_list[index]])
                    self.inter_node_group_list.append(temp)
            self.all_gpu = dist.new_group()

            self.all_inter_node_group = dist.new_group(self.inter_node_list)

            if dist.get_rank() == 0 or dist.get_rank() == 8:
                print('nodes', self.nodes)
                print('intra_node_group_list',self.intra_node_group_list)
                print('inter_node_group',self.inter_node_group_list)
                print('all_inter_node_group', self.inter_node_list)

    def __setstate__(self, state):
        super(SGD_distribute, self).__setstate__(state)


    def step(self, closure=None):

        args = self.args

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            all_grads = []

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if self.compression_buffer==False:
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    # signum
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    else:
                        buf = param_state['momentum_buffer']

                    buf.mul_(momentum).add_((1 - momentum),d_p)
                    d_p.copy_(buf)
                all_grads.append(d_p)

            dev_grads_buckets = _take_tensors(all_grads, self.bucket_size)
            for dev_grads in dev_grads_buckets:
                d_p_new = _flatten_dense_tensors(dev_grads)

                if self.all_reduce:
                    dist.all_reduce(d_p_new) #self.all_gpu, group = 0
                    if self.signum:
                        d_p_new = torch.sign(d_p_new)
                elif self.signum:
                    if self.nodes > 1:
                        if self.compression_buffer:
                            d_p_new, tensor_size = self.compressor.compress(d_p_new)
                        else:
                            d_p_new = torch.sign(d_p_new)

                        if self.local_rank == 0:
                            if dist.get_rank() == 0:
                                d_p_new_list = []
                                for index, inter_node_group in enumerate(self.inter_node_group_list):
                                    d_p_temp = d_p_new.clone()
                                    dist.broadcast(d_p_temp, self.inter_node_list[index + 1], group = inter_node_group)
                                    d_p_new_list.append(d_p_temp)
                            else:
                                dist.broadcast(d_p_new, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1])                                
                                dist.barrier(group = self.all_inter_node_group)

                            if dist.get_rank() == 0:
                                if self.compression_buffer:
                                    d_p_new_list.append(d_p_new) #count itself
                                    d_p_new = self.compressor.majority_vote(d_p_new_list)
                                else:
                                    for d_p_temp in d_p_new_list:
                                        d_p_new.add_(d_p_temp)
                                    d_p_new = d_p_new / self.nodes
                                dist.barrier(group = self.all_inter_node_group)
                            dist.broadcast(d_p_new, 0, group = self.all_inter_node_group)

                        if self.compression_buffer:
                            d_p_new = self.compressor.uncompress(d_p_new, tensor_size)
                else:
                    print('You can not run without signum or all_reduce')

                #unflatten
                dev_grads_new = _unflatten_dense_tensors(d_p_new,dev_grads)
                for grad, reduced in zip(dev_grads, dev_grads_new):
                    grad.copy_(reduced)
            #LARC saving
            self.layer_adaptive_lr = []
            layer_index = 0
            laryer_saving = [1,2,3,23,49,87] #conv1.weight(no bias), bn1.weight, layer1.1.conv1.weight, layer2.1.conv1.weight, layer3.1.conv1.weight, layer4.1.conv1.weight
            ###
            for p in group['params']:
                layer_index += 1
                ###
                '''
                LARC
                This part of code was originally forked from (https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py)
                ''' 
                if args.larc_enable:
                    trust_coefficient = args.larc_trust_coefficient
                    clip = args.larc_clip
                    eps = args.larc_eps
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)
                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = trust_coefficient * (param_norm) / (grad_norm + param_norm * weight_decay + eps)

                        #add adaptive lr saving
                        if layer_index in laryer_saving:
                            self.layer_adaptive_lr.append(adaptive_lr)

                        # clip learning rate for LARC
                        if clip:
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr/group['lr'], 1)

                        else:
                            adaptive_lr = adaptive_lr/group['lr']

                        p.grad.data *= adaptive_lr
                ###


                if self.compression_buffer: #This part of code is temporary
                    if weight_decay != 0:
                        p.grad.data.add_(weight_decay, p.data)
                p.data.add_(-group['lr'], p.grad.data)

        return loss
def calibrateAnalyticGaussianMechanism(epsilon, delta, GS=1, tol = 1.e-12):
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]
    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    GS : upper bound on L2 global sensitivity (GS >= 0)
    tol : error tolerance for binary search (tol > 0)
    Output:
    sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)
        
    sigma = alpha*GS/sqrt(2.0*epsilon)

    return sigma

def sign(grad):
    return [torch.sign(update) for update in grad]

def dpvalue(p):
  return torch.tensor(1) if np.random.random() < p else torch.tensor(-1)

def flatten(grad_update):
    return torch.cat([update.data.view(-1) for update in grad_update])

def unflatten(flattened, normal_shape):
    grad_update = []
    for param in normal_shape:
        n_params = len(param.view(-1))
        grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size())  )
        flattened = flattened[n_params:]

    return grad_update

def dpsign(args, grad):
    sigma=calibrateAnalyticGaussianMechanism(epsilon=args.eps, delta=args.delta, GS=args.l2_norm_clip)
    result=[]
    for update in grad:
        result.append(torch.tensor([dpvalue(Phi(i/sigma)) for i in flatten(update)]).reshape(update.shape))
    return result


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = 'data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)
            else:
                user_groups = cifar_noniid_class(train_dataset, args.num_users, args.class_per_user)
                    
    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = 'data/mnist/'
        else:
            data_dir = 'data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid_class(train_dataset, args.num_users, args.class_per_user)

    return train_dataset, test_dataset, user_groups
def Phi(t):
    return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))
def dpsign( grad):
    sigma=calibrateAnalyticGaussianMechanism(epsilon=1, delta=0.00001, GS=3)
    result=torch.tensor([dpvalue(Phi(i/sigma)) for i in flatten(grad)]).reshape(grad.shape)
    return result

class DPsignSGD(Optimizer):

    def __init__(self, params, lr=0.01, rand_zero=True):
        defaults = dict(lr=lr)
        self.rand_zero = rand_zero
        super(DPsignSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # take sign of gradient
                grad = dpsign(p.grad)

                # randomise zero gradients to ±1
                if self.rand_zero:
                    grad[grad==0] = torch.randint_like(grad[grad==0], low=0, high=2)*2 - 1
                    # assert not (grad==0).any()
                
                # make update
                p.data -= group['lr'] * grad

        return loss