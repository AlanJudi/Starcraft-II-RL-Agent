import argparse
from functools import partial
import os
import shutil
import sys
from pysc2.env import sc2_env
from pysc2.lib import features
from RLAgent.runner import Runner
import ZAgent.Agent as Z
import gui as UI
import RLAgent.RLAgent as RL
import RLAgent.RandomAgent as RA
import RLAgent.DRLAgent as DRLA

from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop

import threading
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

from Networks.environment import SubprocVecEnv, make_sc2env, SingleEnv

# Workaround for pysc2 flags
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['main.py'])



parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents')
parser.add_argument('experiment_id', type=str,
                    help='identifier to store experiment results')
parser.add_argument('--train', action='store_true',
                    help='if false, episode scores are evaluated')
parser.add_argument('--map', type=str, default='Simple64',
                    help='name of SC2 map')
parser.add_argument('--vis', action='store_true',
                    help='render with pygame')
parser.add_argument('--max_windows', type=int, default=1,
                    help='maximum number of visualization windows to open')
parser.add_argument('--res', type=int, default=64,
                    help='screen and minimap resolution')
parser.add_argument('--envs', type=int, default=1,
                    help='number of environments simulated in parallel')
parser.add_argument('--step_mul', type=int, default=5,
                    help='number of game steps per agent step')
parser.add_argument('--steps_per_batch', type=int, default=16,
                    help='number of agent steps when collecting trajectories for a single batch')
parser.add_argument('--discount', type=float, default=0.99,
                    help='discount for future rewards')
parser.add_argument('--iters', type=int, default=-1,
                    help='number of iterations to run (-1 to run forever)')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu device id')
parser.add_argument('--nhwc', action='store_true',
                    help='train fullyConv in NCHW mode')
parser.add_argument('--max_to_keep', type=int, default=5,
                    help='maximum number of checkpoints to keep before discarding older ones')
parser.add_argument('--entropy_weight', type=float, default=1e-3,
                    help='weight of entropy loss')
parser.add_argument('--value_loss_weight', type=float, default=0.5,
                    help='weight of value function loss')

parser.add_argument('--summary_iters', type=int, default=10,
                    help='record training summary after this many iterations')

parser.add_argument('--overwrite', action='store_true',
                    help='overwrite existing experiments (if --train=True)')

parser.add_argument('--save_model', type=str, default=os.path.join('out','models'),
                    help='root directory for checkpoint storage')
parser.add_argument('--lr', type=float, default=7e-4,
                    help='initial learning rate')

parser.add_argument('--save_iters', type=int, default=5000,
                    help='store checkpoint after this many iterations')

parser.add_argument('--summary_path', type=str, default=os.path.join('out','summary'),
                    help='root directory for summary storage')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

ckpt_path = os.path.join(args.save_model, args.experiment_id)

summary_type = 'train' if args.train else 'eval'
summary_path = os.path.join(args.summary_path, args.experiment_id, summary_type)

def _save_if_training(agent, summary_writer):
  if args.train:
    agent.save(ckpt_path)
    summary_writer.flush()
    sys.stdout.flush()

def TrainAgent():

  env_args = dict(
      map_name=args.map,
      step_mul=args.step_mul,
      players=[sc2_env.Agent(sc2_env.Race.terran)],
      game_steps_per_episode=0,
      agent_interface_format=features.AgentInterfaceFormat(
          action_space=actions.ActionSpace.FEATURES,
          
          use_raw_units=True,
          feature_dimensions=features.Dimensions(screen=64, minimap=64),
          use_feature_units=True,
          raw_resolution=args.res,
      ))
  vis_env_args = env_args.copy()
  vis_env_args['visualize'] = args.vis
  num_vis = min(args.envs, args.max_windows)
  env_fns = [partial(make_sc2env, **vis_env_args)] * num_vis
  num_no_vis = args.envs - num_vis
  if num_no_vis > 0:
    env_fns.extend([partial(make_sc2env, **env_args)] * num_no_vis)

  envs = SubprocVecEnv(env_fns)
  sess = tf.compat.v1.Session()
  summary_writer = tf.compat.v1.summary.FileWriter(summary_path)
  network_data_format = 'NHWC' if args.nhwc else 'NCHW'

  agent = DRLA.DRLAgent(
    sess = sess,
    network_data_format=network_data_format,
    value_loss_weight=args.value_loss_weight,
    entropy_weight=args.entropy_weight,
    learning_rate=args.lr,
    max_to_keep=args.max_to_keep)


  runner = Runner(
        envs=envs,
        agent=agent,
        train=args.train,
        summary_writer=summary_writer,
        discount=args.discount,
        n_steps=args.steps_per_batch)

  static_shape_channels = runner.preproc.get_input_channels()
  
  agent.build(static_shape_channels, resolution=args.res)

  if os.path.exists(ckpt_path):
    agent.load(ckpt_path)
  else:
    agent.init()

  runner.reset()
  i = 0
  try:
    while True:
      write_summary = args.train and i % args.summary_iters == 0

      if i > 0 and i % args.save_iters == 0:
        _save_if_training(agent, summary_writer)

      result = runner.run_batch(train_summary=write_summary)

      if write_summary:
        agent_step, loss, summary = result
        summary_writer.add_summary(summary, global_step=agent_step)
        print('iter %d: loss = %f' % (agent_step, loss))

      i += 1

      if 0 <= args.iters <= i:
        break

  except KeyboardInterrupt:
      pass

  _save_if_training(agent, summary_writer)

  envs.close()
  summary_writer.close()

  print('mean score: %f' % runner.get_mean_score())



def init_agents():

    agent1 = RL.RLAgent()
    agent2 = RA.RandomAgent()

    if args.train:
      TrainAgent()
      return


    try:
        with sc2_env.SC2Env(
            map_name=args.map,  # Choose the map
            players=[sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True,
                raw_resolution=args.res,
            ),
            step_mul=args.step_mul,  # How fast it runs the gax`me
            disable_fog=True,  # Too see everything in the minimap
        ) as env:
            # Control both agents instead of one
            run_loop.run_loop([agent1, agent2], env, max_episodes=1000)
    except KeyboardInterrupt:
        env.close()
        pass


def main():
    if args.train and args.overwrite:
      shutil.rmtree(ckpt_path, ignore_errors=True)
      shutil.rmtree(summary_path, ignore_errors=True)


    t1 = threading.Thread(target=init_agents)
    t1.start()
    ui = UI.GUI("RL Agent Starcraft II", "1800x900")

    ui.print_to_tab("Running tests ....", ui.console)

    ui.window.mainloop()
    t1.join()


if __name__ == "__main__":
    main()
