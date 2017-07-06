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


# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--expt', type=str, default='test')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--d_per_iter', type=int, default=1)
train_arg.add_argument('--g_per_iter', type=int, default=1)
train_arg.add_argument('--c_per_iter', type=int, default=1)
train_arg.add_argument('--d_lr', type=float, default=0.001)
train_arg.add_argument('--g_lr', type=float, default=0.001)
train_arg.add_argument('--c_lr', type=float, default=0.001)
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--gamma_d', type=float, default=1)
train_arg.add_argument('--gamma_g', type=float, default=1)
train_arg.add_argument('--lambda_k_d', type=float, default=0.001)
train_arg.add_argument('--lambda_k_g', type=float, default=0.001)
train_arg.add_argument('--lambda_normality_loss', type=float, default=0.001)
train_arg.add_argument('--max_iter', type=int, default=100000)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--training_z', type=str, default='random',
        choices=['preimage', 'random', 'mix'])
train_arg.add_argument('--normality_dist_fn', type=str, default='stein',
        choices=['stein', 'munkres'])
train_arg.add_argument('--interpolation_n', type=int, default=10)

# Output
output_arg = add_argument_group('Output')
output_arg.add_argument('--tensorboard_step', type=int, default=100)
output_arg.add_argument('--checkpoint_step', type=int, default=100)
output_arg.add_argument('--plot_and_print_step', type=int, default=100)
output_arg.add_argument('--plot_z_preimage_step', type=int, default=100)
output_arg.add_argument('--email', type=str2bool, default=False)
output_arg.add_argument('--email_step', type=int, default=5000)

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--d_out_dim', type=int, default=2)
net_arg.add_argument('--d_encoded_dim', type=int, default=3)
net_arg.add_argument('--d_layers_depth', type=int, default=8)
net_arg.add_argument('--d_layers_width', type=int, default=8)
net_arg.add_argument('--z_dim', type=int, default=2)
net_arg.add_argument('--g_out_dim', type=int, default=2)
net_arg.add_argument('--g_layers_depth', type=int, default=8)
net_arg.add_argument('--g_layers_width', type=int, default=8)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='gaussian')
data_arg.add_argument('--real_n', type=int, default=50)
data_arg.add_argument('--real_dim', type=int, default=2)
data_arg.add_argument('--gen_n', type=int, default=50)
data_arg.add_argument('--batch_size', type=int, default=25)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
