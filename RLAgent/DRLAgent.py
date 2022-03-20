import tensorflow as tf
import tf_slim as slim
import os



from Networks.CNN import FullyConv
from Networks.util import safe_log, safe_div
import RLAgent.Agent as Agent
from pysc2.lib import units
import numpy as np


from pysc2.lib.actions import TYPES as ACTION_TYPES

tf.compat.v1.disable_v2_behavior()


  

class DRLAgent(Agent.Agent):
  """DRL agent.
  Run build(...) first, then init() or load(...).
  """
  def __init__(self,
    sess,
    network_data_format='NCHW',
    network_cls=FullyConv,
    value_loss_weight=0.5,
    entropy_weight=1e-3,
    learning_rate=7e-4,
    max_gradient_norm=1.0,
    max_to_keep=5):
    super(DRLAgent, self).__init__()
    self.sess = sess
    self.network_data_format = network_data_format
    self.network_cls = network_cls
    self.value_loss_weight = value_loss_weight
    self.entropy_weight = entropy_weight
    self.learning_rate = learning_rate
    self.max_gradient_norm = max_gradient_norm
    self.train_step = 0
    self.max_to_keep = max_to_keep
    self.new_game()

  # Start the new game and store actions and states for the reinforcement learning
  def new_game(self):
    self.base_top_left = None
    self.previous_state = None
    self.previous_action = None

  def reset(self):
    super(DRLAgent, self).reset()
    self.new_game()

  def build(self, static_shape_channels, resolution, scope=None, reuse=None):
    #with tf.variable_scope(scope, reuse=reuse):
    self._build(static_shape_channels, resolution)
    variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    self.saver = tf.compat.v1.train.Saver(variables, max_to_keep=self.max_to_keep)
    self.init_op = tf.compat.v1.variables_initializer(variables)
    train_summaries = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES, scope=scope)
    self.train_summary_op = tf.compat.v1.summary.merge(train_summaries)

  

  def _build(self, static_shape_channels, resolution):
    """Create tensorflow graph for A2C agent.
    Args:
      static_shape_channels: dict with keys
        {screen, minimap, flat, available_actions}.
      resolution: Integer resolution of screen and minimap.
    """
    channels = static_shape_channels
  
    feature_screen = tf.compat.v1.placeholder(
      tf.compat.v1.float32, [None, resolution, resolution, channels['feature_screen']], 'input_feature_screen')
    feature_minimap = tf.compat.v1.placeholder(
      tf.compat.v1.float32, [None, resolution, resolution, channels['feature_minimap']], 'input_feature_minimap')
    
    flat = tf.compat.v1.placeholder(
      tf.compat.v1.float32, [None, channels['flat']], 'input_flat')
    
    available_actions = tf.compat.v1.placeholder(
      tf.compat.v1.float32, [None, channels['available_actions']], 'input_available_actions')

    advs = tf.compat.v1.placeholder(tf.compat.v1.float32, [None], 'advs')
    returns = tf.compat.v1.placeholder(tf.compat.v1.float32, [None], 'returns')

    self.feature_screen = feature_screen
    self.feature_minimap = feature_minimap
    self.flat = flat
    self.advs = advs
    self.returns = returns
    self.available_actions = available_actions

    policy, value = self.network_cls(data_format=self.network_data_format).build(
        feature_screen, feature_minimap, flat)
    self.policy = policy
    self.value = value

    fn_id = tf.compat.v1.placeholder(tf.compat.v1.int32, [None], 'fn_id')
    arg_ids = {
        k: tf.compat.v1.placeholder (tf.compat.v1.int32, [None], 'arg_{}_id'.format(k.id))
        for k in policy[1].keys()}
    actions = (fn_id, arg_ids)
    self.actions = actions

    log_probs = compute_policy_log_probs(available_actions, policy, actions)

    policy_loss = -tf.compat.v1.reduce_mean(advs * log_probs)
    value_loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(returns - value) / 2.)
    entropy = compute_policy_entropy(available_actions, policy, actions)

    loss = (policy_loss
            + value_loss * self.value_loss_weight
            - entropy * self.entropy_weight)

    tf.compat.v1.summary.scalar('entropy', entropy)
    tf.compat.v1.summary.scalar('loss', loss)
    tf.compat.v1.summary.scalar('loss/policy', policy_loss)
    tf.compat.v1.summary.scalar('loss/value', value_loss)
    tf.compat.v1.summary.scalar('rl/value', tf.compat.v1.reduce_mean(value))
    tf.compat.v1.summary.scalar('rl/returns', tf.compat.v1.reduce_mean(returns))
    tf.compat.v1.summary.scalar('rl/advs', tf.compat.v1.reduce_mean(advs))
    self.loss = loss

    global_step = tf.compat.v1.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(
        self.learning_rate, global_step,
        10000, 0.94)

    opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate,
                                    decay=0.99,
                                    epsilon=1e-5)

    self.train_op = slim.optimize_loss(
        loss=loss,
        global_step=tf.compat.v1.train.get_global_step(),
        optimizer=opt,
        clip_gradients=self.max_gradient_norm,
        learning_rate=None,
        name="train_op")

    self.samples = sample_actions(available_actions, policy)

  def train(self, obs, actions, returns, advs, summary=False):
    """
    Args:
      obs: dict of preprocessed observation arrays, with num_batch elements
        in the first dimensions.
      actions: see `compute_total_log_probs`.
      returns: array of shape [num_batch].
      advs: array of shape [num_batch].
      summary: Whether to return a summary.
    Returns:
      summary: (agent_step, loss, Summary) or None.
    """
    feed_dict = self.get_obs_feed(obs)
    feed_dict.update(self.get_actions_feed(actions))
    feed_dict.update({
        self.returns: returns,
        self.advs: advs})

    ops = [self.train_op, self.loss]

    if summary:
      ops.append(self.train_summary_op)

    res = self.sess.run(ops, feed_dict=feed_dict)
    agent_step = self.train_step
    self.train_step += 1

    if summary:
      return (agent_step, res[1], res[-1])

  def get_mean_score(self):
    return self.cumulative_score / self.episode_counter

  def get_state(self, obs):
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    idle_scvs = [scv for scv in scvs if scv.order_length == 0]
    command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
    supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
    completed_supply_depots = self.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
    barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
    completed_barrackses = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    
    queued_marines = (completed_barrackses[0].order_length if len(completed_barrackses) > 0 else 0)
    
    free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
    can_afford_supply_depot = obs.observation.player.minerals >= 100
    can_afford_barracks = obs.observation.player.minerals >= 150
    can_afford_marine = obs.observation.player.minerals >= 100
    
    enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
    enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
    enemy_command_centers = self.get_enemy_units_by_type(obs, units.Terran.CommandCenter)
    enemy_supply_depots = self.get_enemy_units_by_type(obs, units.Terran.SupplyDepot)
    enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(obs, units.Terran.SupplyDepot)
    enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
    enemy_completed_barrackses = self.get_enemy_completed_units_by_type(obs, units.Terran.Barracks)
    enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)
    
    # Return tuple 
    return (len(command_centers),
            len(scvs),
            len(idle_scvs),
            len(supply_depots),
            len(completed_supply_depots),
            len(barrackses),
            len(completed_barrackses),
            len(marines),
            queued_marines,
            free_supply,
            can_afford_supply_depot,
            can_afford_barracks,
            can_afford_marine,
            len(enemy_command_centers),
            len(enemy_scvs),
            len(enemy_idle_scvs),
            len(enemy_supply_depots),
            len(enemy_completed_supply_depots),
            len(enemy_barrackses),
            len(enemy_completed_barrackses),
            len(enemy_marines))

  # Gets the current state of the game, feeds the state into the NNetwork and the NNetwork provides a policy for choosing an action
  def step(self, obs):
    """
    Args:
      obs: dict of preprocessed observation arrays, with num_batch elements
        in the first dimensions.
    Returns:
      actions: arrays (see `compute_total_log_probs`)
      values: array of shape [num_batch] containing value estimates.
    """
    feed_dict = self.get_obs_feed(obs)
    return self.sess.run([self.samples, self.value], feed_dict=feed_dict)

  

  def get_obs_feed(self, obs):
    return {self.feature_screen: obs['feature_screen'],
            self.feature_minimap: obs['feature_minimap'],
            self.flat: obs['flat'],
            self.available_actions: obs['available_actions']}

  def get_actions_feed(self, actions):
    feed_dict = {self.actions[0]: actions[0]}
    feed_dict.update({v: actions[1][k] for k, v in self.actions[1].items()})
    return feed_dict

  def get_value(self, obs):
    return self.sess.run(
        self.value,
        feed_dict=self.get_obs_feed(obs))

  def init(self):
    self.sess.run(self.init_op)

  def save(self, path, step=None):
    os.makedirs(path, exist_ok=True)
    step = step or self.train_step
    print("Saving agent to %s, step %d" % (path, step))
    ckpt_path = os.path.join(path, 'model.ckpt')
    self.saver.save(self.sess, ckpt_path, global_step=step)

  def load(self, path):
    ckpt = tf.compat.v1.train.get_checkpoint_state(path)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    self.train_step = int(ckpt.model_checkpoint_path.split('-')[-1])
    print("Loaded agent at train_step %d" % self.train_step)



def mask_unavailable_actions(available_actions, fn_pi):
  fn_pi *= available_actions
  fn_pi /= tf.compat.v1.reduce_sum(fn_pi, axis=1, keepdims=True)
  return fn_pi


def compute_policy_entropy(available_actions, policy, actions):
  """Compute total policy entropy.
  Args: (same as compute_policy_log_probs)
  Returns:
    entropy: a scalar float tensor.
  """

  def compute_entropy(probs):
    return -tf.compat.v1.reduce_sum(safe_log(probs) * probs, axis=-1)

  _, arg_ids = actions

  fn_pi, arg_pis = policy
  fn_pi = mask_unavailable_actions(available_actions, fn_pi)
  entropy = tf.compat.v1.reduce_mean(compute_entropy(fn_pi))
  tf.compat.v1.summary.scalar('entropy/fn', entropy)

  for arg_type in arg_ids.keys():
    arg_id = arg_ids[arg_type]
    arg_pi = arg_pis[arg_type]
    batch_mask = tf.compat.v1.to_float(tf.not_equal(arg_id, -1))
    arg_entropy = safe_div(
        tf.compat.v1.reduce_sum(compute_entropy(arg_pi) * batch_mask),
        tf.compat.v1.reduce_sum(batch_mask))
    entropy += arg_entropy
    tf.compat.v1.summary.scalar('used/arg/%s' % arg_type.name,
                      tf.compat.v1.reduce_mean(batch_mask))
    tf.compat.v1.summary.scalar('entropy/arg/%s' % arg_type.name, arg_entropy)

  return entropy


def sample_actions(available_actions, policy):
  """Sample function ids and arguments from a predicted policy."""

  def sample(probs):
    dist = tf.compat.v1.distributions.Categorical(probs=probs)
    return dist.sample()

  fn_pi, arg_pis = policy
  fn_pi = mask_unavailable_actions(available_actions, fn_pi)
  fn_samples = sample(fn_pi)

  arg_samples = dict()
  for arg_type, arg_pi in arg_pis.items():
    arg_samples[arg_type] = sample(arg_pi)

  return fn_samples, arg_samples


def compute_policy_log_probs(available_actions, policy, actions):
  """Compute action log probabilities given predicted policies and selected
  actions.
  Args:
    available_actions: one-hot (in last dimenson) tensor of shape
      [num_batch, NUM_FUNCTIONS].
    policy: [fn_pi, {arg_0: arg_0_pi, ..., arg_n: arg_n_pi}]], where
      each value is a tensor of shape [num_batch, num_params] representing
      probability distributions over the function ids or over discrete
      argument values.
    actions: [fn_ids, {arg_0: arg_0_ids, ..., arg_n: arg_n_ids}], where
      each value is a tensor of shape [num_batch] representing the selected
      argument or actions ids. The argument id will be -1 if the argument is
      not available for a specific (state, action) pair.
  Returns:
    log_prob: a tensor of shape [num_batch]
  """
  def compute_log_probs(probs, labels):
     # Select arbitrary element for unused arguments (log probs will be masked)
    labels = tf.compat.v1.maximum(labels, 0)
    indices = tf.compat.v1.stack([tf.compat.v1.range(tf.compat.v1.shape(labels)[0]), labels], axis=1)
    return safe_log(tf.compat.v1.gather_nd(probs, indices)) # TODO tf.log should suffice

  fn_id, arg_ids = actions
  fn_pi, arg_pis = policy
  fn_pi = mask_unavailable_actions(available_actions, fn_pi) # TODO: this should be unneccessary
  fn_log_prob = compute_log_probs(fn_pi, fn_id)
  tf.compat.v1.summary.scalar('log_prob/fn', tf.compat.v1.reduce_mean(fn_log_prob))

  log_prob = fn_log_prob
  for arg_type in arg_ids.keys():
    arg_id = arg_ids[arg_type]
    arg_pi = arg_pis[arg_type]
    arg_log_prob = compute_log_probs(arg_pi, arg_id)
    arg_log_prob *= tf.compat.v1.to_float(tf.compat.v1.not_equal(arg_id, -1))
    log_prob += arg_log_prob
    tf.compat.v1.summary.scalar('log_prob/arg/%s' % arg_type.name,
                      tf.compat.v1.reduce_mean(arg_log_prob))

  return log_prob