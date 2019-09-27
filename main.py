import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from world_model import *
from config import argparser
from utils import make_env as env_fn, printstar

g_env_model = None

def main(config):
    global env_fn
    env = env_fn()()
    env.reset()
    
    action_dim = 5
    ob_shape = env.observation_space.shape
    world_model_path = os.path.expanduser(os.path.join(config.model_dir, config.world_model_type + "_" + config.world_model_path))

    if(config.train_world_model):      
        env_model = EnvModel(ob_shape, action_dim, config)
        if(not os.path.exists(world_model_path)):
            os.mkdir(world_model_path)
        printstar("Training World Model")
        env_model.train(world_model_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if(config.eval_world_model):     
            env_model = cached_world_model(sess, ob_shape, action_dim, config, world_model_path + '/env_model.ckpt')
            evaluate_world_model(env, sess, env_model, config)        

def evaluate_world_model(env, sess, env_model, config, policy=None):
    printstar("Testing World Model")
    obs = env.reset()
    for t in range(config.max_eval_iters):
        if(policy is None):
                action = env.action_space.sample()
        else:
            action = policy(obs)

        next_pred_ob = env_model.imagine(sess, obs, action)
        imgplot = plt.imshow(next_pred_ob)
        plt.savefig('./figs/world_model_eval.png')

        env.render()
        obs, reward, dones, info = env.step(action)
        inp = input("Press 0 to exit : ")
        if(inp == "0"):
            break

if __name__ == '__main__':
    config = argparser()
    mp.set_start_method('spawn', force=True)
    main(config)
