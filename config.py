#-*- coding: utf-8 -*-
import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--expt', type=str, default='test')
misc_arg.add_argument('--email', type=str2bool, default=False)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--train', type=str2bool, default=True)
train_arg.add_argument('--d_lr', type=float, default=0.001)
train_arg.add_argument('--g_lr', type=float, default=0.001)
train_arg.add_argument('--c_lr', type=float, default=0.001)
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--gamma_d', type=float, default=0.5)
train_arg.add_argument('--gamma_g', type=float, default=0.5)
train_arg.add_argument('--lambda_k_d', type=float, default=0.001)
train_arg.add_argument('--lambda_k_g', type=float, default=0.001)
train_arg.add_argument('--max_iter', type=int, default=500000)
train_arg.add_argument('--optimizer', type=str, default='adam')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--d_out_dim', type=int, default=2)
net_arg.add_argument('--d_bottleneck_dim', type=int, default=3)
net_arg.add_argument('--d_layers_depth', type=int, default=5)
net_arg.add_argument('--d_layers_width', type=int, default=5)
net_arg.add_argument('--z_dim', type=int, default=2)
net_arg.add_argument('--g_out_dim', type=int, default=2)
net_arg.add_argument('--g_layers_depth', type=int, default=5)
net_arg.add_argument('--g_layers_width', type=int, default=5)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='gaussian')
data_arg.add_argument('--real_n', type=int, default=1000)
data_arg.add_argument('--real_dim', type=int, default=2)
data_arg.add_argument('--gen_n', type=int, default=1000)
data_arg.add_argument('--d_batch_size', type=int, default=25)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
