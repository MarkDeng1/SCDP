import torch
import torch.optim as optim
import copy
import math
import numpy as np
from quantization import LatticeQuantization, ScalarQuantization
from configurations import args_parser
import concurrent.futures
# from privacy import Privacy
from integer_convert import binary_convert,process_tensors

def federated_setup(global_model, train_data, args):
    # create a dict of dict s (local users), i.e. {'1': {'data':..., 'model':..., 'opt':...}, ...}
    indexes = torch.randperm(len(train_data))
    user_data_len = math.floor(len(train_data) / args.num_users)
    local_models = {}
    for user_idx in range(args.num_users):
        user = {'data': torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data,
                                    indexes[user_idx * user_data_len:(user_idx + 1) * user_data_len]),
            batch_size=args.train_batch_size, shuffle=True),
            'model': copy.deepcopy(global_model)}
        user['opt'] = optim.SGD(user['model'].parameters(), lr=args.lr,
                                momentum=args.momentum) if args.optimizer == 'sgd' \
            else optim.Adam(user['model'].parameters(), lr=args.lr)
        # if args.lr_scheduler:
        #     user['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(user['opt'], patience=10, factor=0.1, verbose=True)
        local_models[user_idx] = user
    return local_models


def distribute_model(local_models, global_model):
    for user_idx in range(len(local_models)):
        local_models[user_idx]['model'].load_state_dict(copy.deepcopy(global_model.state_dict()))



def update_state_dict(state_dict, local_models, mechanism, binary_convert, process_tensors, args):
    for key in state_dict.keys():
        local_weights_average = torch.zeros_like(state_dict[key])
        list_local_weights = []
        for user_idx in range(0, len(local_models)):
            local_weights_orig = local_models[user_idx]['model'].state_dict()[key] - state_dict[key]
            if args.privacy and args.quantization:
                local_weights = mechanism(local_weights_orig)
                if key.startswith('linear'):
                    local_weights_bi = binary_convert(local_weights, p=0.98)
                    list_local_weights.append(local_weights_bi)
                else:
                    local_weights_average += local_weights
                    
            elif args.quantization:
                local_weights = mechanism(local_weights_orig)
                local_weights_average += local_weights
            else: 
                local_weights_average += local_weights_orig
        if args.privacy and args.quantization:
            if key.startswith('linear'):
                value = state_dict[key] + process_tensors(list_local_weights, p=0.98).to(args.device)
                state_dict[key] = value.detach().clone()
            else:
                state_dict[key] += (local_weights_average / len(local_models)).to(state_dict[key].dtype)
                
        else:
            state_dict[key] += (local_weights_average / len(local_models)).to(state_dict[key].dtype)
def aggregate_models(local_models, global_model, mechanism):  # FeaAvg
    #定义平均值的函数
    # mean = lambda x: sum(x) / len(x)
    #复制参数包括对应的号,  其结构类似为
#     {
#     'conv1.weight': tensor([...]),
#     'conv1.bias': tensor([...]),
#     'fc1.weight': tensor([...]),
#     'fc1.bias': tensor([...]),
#     ...
# }
    args = args_parser()
    state_dict = copy.deepcopy(global_model.state_dict())
    SNR_layers = []
    
    update_state_dict(state_dict, local_models, mechanism, binary_convert, process_tensors, args)
    # for key in state_dict.keys():
    #     local_weights_average = torch.zeros_like(state_dict[key])
    #     SNR_users = []
    #     list_local_weights = []
    #     for user_idx in range(0, len(local_models)):
    #         local_weights_orig = local_models[user_idx]['model'].state_dict()[key] - state_dict[key]
    #         local_weights = mechanism(local_weights_orig)
    #         local_weights_bi = binary_convert(local_weights, p = 0.98)
    #         list_local_weights.append(local_weights_bi)
    #         # SNR_users.append(torch.var(local_weights_orig) / torch.var(local_weights_orig - local_weights))
    #         local_weights_average += local_weights
    #     # SNR_layers.append(mean(SNR_users))
        
    #     value = state_dict[key] + process_tensors(list_local_weights, p = 0.98).to(args.device)
    #     state_dict[key] = value.detach().clone()
    # global_model.load_state_dict(copy.deepcopy(state_dict))
    if args.privacy:
        global_model.load_state_dict(state_dict)
    else:
        global_model.load_state_dict(copy.deepcopy(state_dict))
    return 1 # mean(SNR_layers)

class Quantize:  # Privacy Quantization class
    def __init__(self, args):
        self.vec_normalization = args.vec_normalization
        dither_var = None
        if args.quantization:
            if args.lattice_dim > 1:
                self.quantizer = LatticeQuantization(args)
                dither_var = self.quantizer.P0_cov
            else:
                self.quantizer = ScalarQuantization(args)
                dither_var = (self.quantizer.delta ** 2) / 12
        else:
            self.quantizer = None

    def divide_into_blocks(self, input, dim=2):
        # Zero pad if needed
        modulo = len(input) % dim
        if modulo:
            pad_with = dim - modulo
            input_vec = torch.cat((input, torch.zeros(pad_with).to(input.dtype).to(input.device)))
        else:
            pad_with = 0
        # 把输入自动变换维度
        input_vec = input.view(dim, -1)  # divide input into blocks
        return input_vec, pad_with,

    def __call__(self, input):
        original_shape = input.shape

        if input.numel() != 1 and input.numel() != 0:
        # 把input一维展开
            input = input.view(-1)
            if self.vec_normalization:  # normalize
                input, pad_with = self.divide_into_blocks(input)
            # 计算输入input的均值和标准差
            mean = torch.mean(input, dim=-1, keepdim=True)
            std = torch.norm(input - mean) / (input.shape[-1] ** 0.5)
            std = 3 * std
            input = (input - mean) / std


            if self.quantizer is not None:
                input = self.quantizer(input)

            # denormalize
            input = (input * std) + mean

            if self.vec_normalization:
                input = input.view(-1)[:-pad_with] if pad_with else input  # remove zero padding

            input = input.reshape(original_shape)

        return input
