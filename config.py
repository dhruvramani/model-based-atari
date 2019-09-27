import os
import argparse

def str2bool(v):
    return v.lower() == 'true'

def str2list(v):
    if not v:
        return v
    else:
        return [v_ for v_ in v.split(',')]

def argparser():
    parser = argparse.ArgumentParser("Model-Based Reinforcement Learning for Atari",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train_world_model', type=str2bool, default=True)
    parser.add_argument('--eval_world_model', type=str2bool, default=True)

    parser.add_argument('--num_rewards', type=int, default=1)
    parser.add_argument('--n_envs', type=int, default=16)
    parser.add_argument('--is_policy', type=str2bool, default=False)
    parser.add_argument('--has_rewards', type=str2bool, default=False)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--max_ep_len', type=int, default=500000)
    parser.add_argument('--l2_clip', type=float, default=0.0)
    parser.add_argument('--softmax_clip', type=float, default=0.03)
    parser.add_argument('--reward_coeff', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--activation_fn', type=str, default='relu', choices=['relu', 'tanh'])

    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--world_model_path', type=str, default="CarRacingWorldModel")
    
    parser.add_argument('--total_timesteps', type=int, default=int(1e6))
    parser.add_argument('--max_eval_iters', type=int, default=int(1e3))
    
    parser.add_argument('--render', type=str2bool, default=True, help='Render frames')
    parser.add_argument('--debug', type=str2bool, default=False, help='See debugging info')

    args = parser.parse_args()
    return args
