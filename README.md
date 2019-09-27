# Model-Based Reinforcement Learning for Atari

An easy to understand/use implementation of the deterministic world model presented in the paper ["Model-Based Reinforcement Learning for Atari"](https://arxiv.org/abs/1903.00374) as compared to the [official implementation](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/rl). Can be used to incorporate the model easily in your experiments for Atari or other environments with image-based state space. Currently consists of the most basic deterministic model presented in the paper. 

## Installation & Setup
 
The code is written in [Tensorflow](http://tensorflow.org/), so get that if you don't have it already. To run it on your environment, change the `make_env/_thunk` function in `utils.py`. The rest of the parameters of the network/experiment can be changed in `config.py`.  

## Edits & Additions

As compared to the original code, we just take the single frame as the input. This was done to efficiently generate rollouts and trajectories. We train the world model on observations of multiple agents exploring the same environment differently. This is done by acting randomly, however a policy can be used to generate the data by modiflying the function `generate_data` in `world_model.py`. The number of agents can be changed by changing `n_envs` in `config.py`. The next observation can be predicted easily by calling `EnvModel.imagine()`.

# References
[Original Tensor2Tensor code](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/rl)
[I2A Code](https://github.com/gitlimlab/i2a-tf)

Thank me! I just saved around 3 days of your time. T2T's code-base is huuuuuge, and very hard to modify. :heart:
