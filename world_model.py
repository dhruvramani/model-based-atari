import os
import gym 
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from utils import *

g_env_model = None
def cached_world_model(sess, ob_shape, action_dim, config, path):
    global g_env_model
    if g_env_model is None:
        old_val = config.n_envs
        config.n_envs = 1
        g_env_model = EnvModel(ob_shape, action_dim, config)
        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='env_model')
        loader = tf.train.Saver(var_list=save_vars)
        loader.restore(sess, path)
        printstar('World Model Restored')
        config.n_envs = old_val

    return g_env_model

def inject_additional_input(layer, inputs, name, mode="multi_additive"):
  """Injects the additional input into the layer.

  Args:
    layer: layer that the input should be injected to.
    inputs: inputs to be injected.
    name: TF scope name.
    mode: how the infor should be added to the layer:
      "concat" concats as additional channels.
      "multiplicative" broadcasts inputs and multiply them to the channels.
      "multi_additive" broadcasts inputs and multiply and add to the channels.

  Returns:
    updated layer.

  Raises:
    ValueError: in case of unknown mode.
  """
  layer_shape = shape_list(layer)
  input_shape = shape_list(inputs)
  zeros_mask = tf.zeros(layer_shape, dtype=tf.float32)
  if mode == "concat":
    emb = common_video.encode_to_shape(inputs, layer_shape, name)
    layer = tf.concat(values=[layer, emb], axis=-1)
  elif mode == "multiplicative":
    filters = layer_shape[-1]
    input_reshaped = tf.reshape(inputs, [-1, 1, 1, input_shape[-1]])
    input_mask = tf.layers.dense(input_reshaped, filters, name=name)
    input_broad = input_mask + zeros_mask
    layer *= input_broad
  elif mode == "multi_additive":
    filters = layer_shape[-1]
    input_reshaped = tf.reshape(inputs, [-1, 1, 1, input_shape[-1]])
    input_mul = tf.layers.dense(input_reshaped, filters, name=name + "_mul")
    layer *= tf.nn.sigmoid(input_mul)
    input_add = tf.layers.dense(input_reshaped, filters, name=name + "_add")
    layer += input_add
  else:
    raise ValueError("Unknown injection mode: %s" % mode)

  return layer

class EnvModel(object):
    def __init__(self, obs_shape, action_dim, config):
        
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.config = config

        self.hidden_size = config.hidden_size
        self.layers = config.n_layers
        self.dropout_p = config.dropout_p
        
        if(config.activation_fn == 'relu'):
            self.activation_fn = tf.nn.relu
        elif (config.activation_fn == 'tanh'):
            self.activation_fn = tf.nn.tanh
        
        self.l2_clip = config.l2_clip
        self.softmax_clip = config.softmax_clip
        self.reward_coeff = config.reward_coeff
        self.n_envs = config.n_envs
        self.max_ep_len = config.max_ep_len
        self.log_interval = config.log_interval

        self.is_policy = config.is_policy
        self.has_rewards = config.has_rewards
        self.num_rewards = config.num_rewards

        self.width, self.height, self.depth = self.obs_shape

        self.states_ph = tf.placeholder(tf.float32, [None, self.width, self.height, self.depth])
        self.actions_ph = tf.placeholder(tf.uint8, [None, 1])
        self.actions_oph = tf.one_hot(self.actions_ph, depth=action_dim)
        self.target_states = tf.placeholder(tf.float32, [None, self.width, self.height, self.depth])
        if(self.has_rewards):
            self.target_rewards = tf.placeholder(tf.uint8, [None, self.num_rewards])
        
        # NOTE - Implement policy and value parts later
        with tf.variable_scope("env_model"):
            self.state_pred, self.reward_pred, _, _ = self.network()

        # NOTE - Change this maybe to video_l2_loss
        self.state_loss = tf.math.maximum(tf.reduce_sum(tf.pow(self.state_pred - self.target_states, 2)), self.l2_clip)
        self.loss = self.state_loss

        if(self.has_rewards):
            self.reward_loss = tf.math.maximum(tf.reduce_mean(tf.losses.softmax_cross_entropy(self.tw_one_hot, self.reward_pred)), self.softmax_clip)
            self.loss = self.loss + (self.reward_coeff * self.reward_loss)

        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

        tf.summary.scalar('loss', self.loss)
        if(self.has_rewards):
            tf.summary.scalar('image_loss', self.state_loss)
            tf.summary.scalar('reward_loss', self.reward_loss)

    def generate_data(self, envs):
        states = envs.reset()
        for frame_idx in range(self.max_ep_len):
            states = states.reshape(self.n_envs, self.width, self.height, self.depth)
            if(self.n_envs == 1):
                actions = envs.action_space.sample()
            else:
                actions = [envs.action_space.sample() for _ in range(self.n_envs)]
            next_states, rewards, dones, _ = envs.step(actions)
            next_states = next_states.reshape(self.n_envs, self.width, self.height, self.depth)

            yield frame_idx, states, actions, rewards, next_states, dones
            states = next_states
            if(self.n_envs == 1 and dones == True):
                states = envs.reset()

    def network(self):
        def middle_network(layer):
            x = layer
            kernel1 = (3, 3)
            filters = shape_list(x)[-1]
            for i in range(2):
              with tf.variable_scope("layer%d" % i):
                y = tf.nn.dropout(x, 1.0 - 0.5)
                y = tf.layers.conv2d(y, filters, kernel1, activation=self.activation_fn,
                                     strides=(1, 1), padding="SAME")
                if i == 0:
                  x = y
                else:
                  x = layer_norm(x + y)
            return x

        batch_size = tf.shape(self.states_ph)[0]

        filters = self.hidden_size
        kernel2 = (4, 4)
        action = self.actions_oph

        # Normalize states
        if(self.n_envs > 1):
            states = [standardize_images(self.states_ph[i, :, :, :]) for i in range(self.n_envs)]
            stacked_states = tf.stack(states)
        else:
            stacked_states = standardize_images(self.states_ph)
        inputs_shape = shape_list(stacked_states)

        # Using non-zero bias initializer below for edge cases of uniform inputs.
        x = tf.layers.dense(
            stacked_states, filters, name="inputs_embed",
            bias_initializer=tf.random_normal_initializer(stddev=0.01))
        x = add_timing_signal_nd(x)

        # Down-stride.
        layer_inputs = [x]
        for i in range(self.layers):
          with tf.variable_scope("downstride%d" % i):
            layer_inputs.append(x)
            x = tf.nn.dropout(x, 1.0 - self.dropout_p)
            x = make_even_size(x)
            if i < 2:
              filters *= 2
            x = add_timing_signal_nd(x)
            x = tf.layers.conv2d(x, filters, kernel2, activation=self.activation_fn,
                                 strides=(2, 2), padding="SAME")
            x = layer_norm(x)

        if self.is_policy:
          with tf.variable_scope("policy"):
            x_flat = tf.layers.flatten(x)
            policy_pred = tf.layers.dense(x_flat, self.action_dim)
            value_pred = tf.layers.dense(x_flat, 1)
            value_pred = tf.squeeze(value_pred, axis=-1)
        else:
          policy_pred, value_pred = None, None

        x = inject_additional_input(x, action, "action_enc", "multi_additive")

        # Inject latent if present. Only for stochastic models.
        target_states = standardize_images(self.target_states)

        x_mid = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = middle_network(x)

        # Up-convolve.
        layer_inputs = list(reversed(layer_inputs))
        for i in range(self.layers):
          with tf.variable_scope("upstride%d" % i):
            x = tf.nn.dropout(x, 1.0 - 0.1)
            if i >= self.layers - 2:
              filters //= 2
            x = tf.layers.conv2d_transpose(
                x, filters, kernel2, activation=self.activation_fn,
                strides=(2, 2), padding="SAME")
            y = layer_inputs[i]
            shape = shape_list(y)
            x = x[:, :shape[1], :shape[2], :]
            x = layer_norm(x + y)
            x = add_timing_signal_nd(x)

        # Cut down to original size.
        x = x[:, :inputs_shape[1], :inputs_shape[2], :]
        x_fin = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        
        x = tf.layers.dense(x, self.depth, name="logits")

        reward_pred = None
        if self.has_rewards:
          # Reward prediction based on middle and final logits.
          reward_pred = tf.concat([x_mid, x_fin], axis=-1)
          reward_pred = tf.nn.relu(tf.layers.dense(
              reward_pred, 128, name="reward_pred"))
          reward_pred = tf.squeeze(reward_pred, axis=1)  # Remove extra dims
          reward_pred = tf.squeeze(reward_pred, axis=1)  # Remove extra dims

        return x, reward_pred, policy_pred, value_pred

    def imagine(self, sess, obs, action):
        action = np.array(action)
        action = np.reshape(action, (1, 1))
        obs = obs.reshape(1, self.width, self.height, self.depth)    
        next_pred_ob = sess.run(self.state_pred, feed_dict={self.states_ph : obs, self.actions_ph : action})
        next_pred_ob = next_pred_ob.reshape(self.width, self.height, self.depth)
        next_pred_ob = np.rint(next_pred_ob)
        return next_pred_ob

    def train(self, world_model_path):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            losses = []
            all_rewards = []
            save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='env_model')
            saver = tf.train.Saver(var_list=save_vars)

            train_writer = tf.summary.FileWriter('./env_logs/train/', graph=sess.graph)
            summary_op = tf.summary.merge_all()

            if(self.n_envs == 1):
                envs = make_env()()
            else:
                envs = [make_env() for i in range(self.n_envs)]
                envs = SubprocVecEnv(envs)

            for idx, states, actions, rewards, next_states, dones in tqdm(
                self.generate_data(envs), total=self.max_ep_len):
                actions = np.array(actions)
                actions = np.reshape(actions, (-1, 1))

                if(self.has_rewards):
                    target_reward = reward_to_target(rewards)
                    loss, reward_loss, state_loss, summary, _ = sess.run([self.loss, self.reward_loss, self.state_loss,
                        summary_op, self.opt], feed_dict={
                        self.states_ph: states,
                        self.actions_ph: actions,
                        self.target_states: next_states,
                        self.target_rewards: target_reward
                    })
                else :
                    loss, summary, _ = sess.run([self.loss, summary_op, self.opt], feed_dict={
                        self.states_ph: states,
                        self.actions_ph: actions,
                        self.target_states: next_states,
                    })

                if idx % self.log_interval == 0:
                    if(self.has_rewards):
                        print('%i => Loss : %.4f, Reward Loss : %.4f, Image Loss : %.4f' % (idx, loss, reward_loss, state_loss))
                    else :
                        print('%i => Loss : %.4f' % (idx, loss))
                    saver.save(sess, '{}/env_model.ckpt'.format(world_model_path))
                    print('Environment model saved')

                train_writer.add_summary(summary, idx)
            envs.close()
