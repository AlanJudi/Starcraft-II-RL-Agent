# Adapted from
# https://github.com/pekaalto/sc2aibot/blob/master/common/multienv.py

from multiprocessing import Pipe, Process
import numpy as np

from pysc2.env import sc2_env

from Networks.VecEnv import VecEnv, clear_mpi_env_vars




# below (worker, CloudpickleWrapper, ProcessEnv) copied from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
# with some sc2 specific modifications
def worker(remote, env_fn_wrappers):
  """
  Handling the:
  action -> [action] and  [timestep] -> timestep
  multi-player conversions here
  """

  env = env_fn_wrappers.x()
  try:
    while True:
      cmd, action = remote.recv()
      if cmd == 'step':
        remote.send(env.step([action]))
        #timesteps = env.step([action])
        #remote.send(timesteps[0])
      elif cmd == 'reset':
        remote.send(env.reset())
        #timesteps = env.reset()
        #remote.send(timesteps[0])

      elif cmd == 'render':
        remote.send(env.render(mode='rgb_array'))
      elif cmd == 'close':
        remote.close()
        break
        #remote.close()
        #break
      elif cmd == 'get_spaces_spec':
        remote.send(CloudpickleWrapper((env.observation_spec(), env.action_spec())))
      elif cmd == 'observation_spec':
        spec = env.observation_spec
        remote.send(spec)
      else:
        raise NotImplementedError
  except KeyboardInterrupt:
    print('Sub process worker: got KeyboardInterrupt')
  finally:
    env.close()



class CloudpickleWrapper(object):
  """
  Uses cloudpickle to serialize contents (otherwise multiprocessing tries
  to use pickle).
  """

  def __init__(self, x):
    self.x = x

  def __getstate__(self):
    import cloudpickle
    return cloudpickle.dumps(self.x)

  def __setstate__(self, ob):
    import pickle
    self.x = pickle.loads(ob)


class ProcessEnv(VecEnv):
  ''' Sub process environments

  Args:
      env_fns: array with pysc2 env functions
  
  '''
  def __init__(self, env_fns, in_series=1):
    self.in_series = in_series
    self.waiting = False
    self.closed = False

    envs_n = len(env_fns)
    assert envs_n % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
    
    self.remotes_n = envs_n // in_series
    
    self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.remotes_n)])
    self.processes = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
               for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
    
    for process in self.processes:
      #start process
      process.daemon = True  # if the main process crashes, we should not cause things to hang
      with clear_mpi_env_vars():
        process.start()

  
    self.envs_n = envs_n
    self.remotes[0].send(('get_spaces_spec', None))
    observation_spec, action_spec= self.remotes[0].recv().x
    self.viewer = None
    VecEnv.__init__(self, envs_n, observation_spec, action_spec)


  def step_async(self, actions):
    self._assert_not_closed()

    actions = actions or [None, None] * self.n_envs
    for remote, action in zip(self.remotes, actions):
      remote.send(('step', action))
    self.waiting = True

  def step_wait(self):
    self._assert_not_closed()
    results = [remote.recv() for remote in self.remotes]
    results = _flatten_list(results)
    self.waiting = False

    return results


  def close_extras(self):
    self.closed = True
    if self.waiting:
      for remote in self.remotes:
        remote.recv()
    for remote in self.remotes:
      remote.send(('close', None))
    for p in self.processes:
      p.join()

  def get_images(self):
    self._assert_not_closed()
    for pipe in self.remotes:
        pipe.send(('render', None))
    imgs = [pipe.recv() for pipe in self.remotes]
    imgs = _flatten_list(imgs)
    return imgs

  def reset(self):
    ''' reset enviornment'''
    self._assert_not_closed()
    for remote in self.remotes:
      remote.send(('reset', None))

    obs = [remote.recv() for remote in self.remotes]
    obs = _flatten_list(obs)
    return _flatten_obs(obs)

  def close(self):
    ''' Close environments '''
    for remote in self.remotes:
      remote.send(('close', None))
    for p in self.processes:
      p.join()

  def _assert_not_closed(self):
    assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

  def __del__(self):
    if not self.closed:
      self.close()


def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)

def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]

def make_sc2env(**kwargs):
  ''' make starcraft environment'''
  env = sc2_env.SC2Env(**kwargs)
  return env