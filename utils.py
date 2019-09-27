# Code is from OpenAI Baseline and Tensor2Tensor

import itertools
import numpy as np
from gym.envs.box2d import CarRacing 
import multiprocessing as mp

def printstar(string, num_stars=50):
    print("*" * num_stars)
    print(string)
    print("*" * num_stars)

def make_env():
    def _thunk():
        env = CarRacing(grayscale=0, show_info_panel=0, discretize_actions="hard", frames_per_state=1, num_lanes=1, num_tracks=1)
        return env
    return _thunk

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
        
class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(nenvs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        if(type(actions) == int):
            for remote in self.remotes:
                remote.send(('step', actions))
        else:
            for remote, action in zip(self.remotes, actions):
                remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def __len__(self):
        return self.nenvs



def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i, dim in enumerate(static):
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

def to_float(x):
  """Cast x to float; created because tf.to_float is deprecated."""
  return tf.cast(x, tf.float32)

def cast_like(x, y):
  """Cast x to y's dtype, if necessary."""
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)

  if x.dtype.base_dtype == y.dtype.base_dtype:
    return x

  cast_x = tf.cast(x, y.dtype)
  if cast_x.device != x.device:
    x_name = "(eager Tensor)"
    try:
      x_name = x.name
    except AttributeError:
      pass
    tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
                       x.device, cast_x.device)
  return cast_x

def layer_norm_vars(filters):
  """Create Variables for layer norm."""
  scale = tf.get_variable(
      "layer_norm_scale", [filters], initializer=tf.ones_initializer())
  bias = tf.get_variable(
      "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
  return scale, bias


def layer_norm_compute(x, epsilon, scale, bias, layer_collection=None):
  """Layer norm raw computation."""

  # Save these before they get converted to tensors by the casting below
  params = (scale, bias)

  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
  variance = tf.reduce_mean(
      tf.squared_difference(x, mean), axis=[-1], keepdims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)

  output = norm_x * scale + bias


  return output

def layer_norm(x,
               filters=None,
               epsilon=1e-6,
               name=None,
               reuse=None,
               layer_collection=None):
  """Layer normalize the tensor x, averaging over the last dimension."""
  if filters is None:
    filters = shape_list(x)[-1]
  with tf.variable_scope(
      name, default_name="layer_norm", values=[x], reuse=reuse):
    scale, bias = layer_norm_vars(filters)
    return layer_norm_compute(x, epsilon, scale, bias,
                              layer_collection=layer_collection)

def standardize_images(x):
  """Image standardization on batches and videos."""
  with tf.name_scope("standardize_images", values=[x]):
    x_shape = shape_list(x)
    x = to_float(tf.reshape(x, [-1] + x_shape[-3:]))
    x_mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    x_variance = tf.reduce_mean(
        tf.squared_difference(x, x_mean), axis=[1, 2], keepdims=True)
    num_pixels = to_float(x_shape[-2] * x_shape[-3])
    x = (x - x_mean) / tf.maximum(tf.sqrt(x_variance), tf.rsqrt(num_pixels))
    return tf.reshape(x, x_shape)

def pad_to_same_length(x, y, final_length_divisible_by=1, axis=1):
  """Pad tensors x and y on axis 1 so that they have the same length."""
  if axis not in [1, 2]:
    raise ValueError("Only axis=1 and axis=2 supported for now.")
  with tf.name_scope("pad_to_same_length", values=[x, y]):
    x_length = shape_list(x)[axis]
    y_length = shape_list(y)[axis]
    if (isinstance(x_length, int) and isinstance(y_length, int) and
        x_length == y_length and final_length_divisible_by == 1):
      return x, y
    max_length = tf.maximum(x_length, y_length)
    if final_length_divisible_by > 1:
      # Find the nearest larger-or-equal integer divisible by given number.
      max_length += final_length_divisible_by - 1
      max_length //= final_length_divisible_by
      max_length *= final_length_divisible_by
    length_diff1 = max_length - x_length
    length_diff2 = max_length - y_length

    def padding_list(length_diff, arg):
      if axis == 1:
        return [[[0, 0], [0, length_diff]],
                tf.zeros([tf.rank(arg) - 2, 2], dtype=tf.int32)]
      return [[[0, 0], [0, 0], [0, length_diff]],
              tf.zeros([tf.rank(arg) - 3, 2], dtype=tf.int32)]

    paddings1 = tf.concat(padding_list(length_diff1, x), axis=0)
    paddings2 = tf.concat(padding_list(length_diff2, y), axis=0)
    res_x = tf.pad(x, paddings1)
    res_y = tf.pad(y, paddings2)
    # Static shapes are the same except for axis=1.
    x_shape = x.shape.as_list()
    x_shape[axis] = None
    res_x.set_shape(x_shape)
    y_shape = y.shape.as_list()
    y_shape[axis] = None
    res_y.set_shape(y_shape)
    return res_x, res_y

def make_even_size(x):
  """Pad x to be even-sized on axis 1 and 2, but only if necessary."""
  x_shape = x.get_shape().as_list()
  assert len(x_shape) > 2, "Only 3+-dimensional tensors supported."
  shape = [dim if dim is not None else -1 for dim in x_shape]
  new_shape = x_shape  # To make sure constant shapes remain constant.
  if x_shape[1] is not None:
    new_shape[1] = 2 * int(math.ceil(x_shape[1] * 0.5))
  if x_shape[2] is not None:
    new_shape[2] = 2 * int(math.ceil(x_shape[2] * 0.5))
  if shape[1] % 2 == 0 and shape[2] % 2 == 0:
    return x
  if shape[1] % 2 == 0:
    x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=2)
    x.set_shape(new_shape)
    return x
  if shape[2] % 2 == 0:
    x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=1)
    x.set_shape(new_shape)
    return x
  x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=1)
  x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=2)
  x.set_shape(new_shape)
  return x


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase in one of the positional dimensions.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(a+b) and cos(a+b) can be
  expressed in terms of b, sin(a) and cos(a).

  x is a Tensor with n "positional" dimensions, e.g. one dimension for a
  sequence or two dimensions for an image

  We use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels // (n * 2). For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    x: a Tensor with shape [batch, d1 ... dn, channels]
    min_timescale: a float
    max_timescale: a float

  Returns:
    a Tensor the same shape as x.
  """
  num_dims = len(x.get_shape().as_list()) - 2
  channels = shape_list(x)[-1]
  num_timescales = channels // (num_dims * 2)
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  for dim in range(num_dims):
    length = shape_list(x)[dim + 1]
    position = tf.to_float(tf.range(length))
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
        inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    prepad = dim * 2 * num_timescales
    postpad = channels - (dim + 1) * 2 * num_timescales
    signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
    for _ in range(1 + dim):
      signal = tf.expand_dims(signal, 0)
    for _ in range(num_dims - 1 - dim):
      signal = tf.expand_dims(signal, -2)
    x += signal
  return x

