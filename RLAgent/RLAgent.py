import RLAgent.Agent as Agent
import QLearn.QTable
from pysc2.lib import units

# Similar to Random Agent but also initialize the QLearning Table to know which actions can perform and learn
class RLAgent(Agent.Agent):
  def __init__(self):
    super(RLAgent, self).__init__()
    self.qtable = QLearn.QTable.QLearningTable(self.actions)
    self.new_game()

  def reset(self):
    super(RLAgent, self).reset()
    print(self.qtable.q_table)
    self.qtable.count += 1
    if self.qtable.count == 100:
      self.qtable.q_table.to_excel(r'QLearningTable.xlsx', sheet_name='QLearningTable', index = False)
    self.new_game()
  
  # Start the new game and store actions and states for the reinforcement learning
  def new_game(self):
    self.base_top_left = None
    self.previous_state = None
    self.previous_action = None

  # Takes all the values of the game we find important and then returning those in a tuple to feed into our machine learning algorithm
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

  # Gets the current state of the game, feeds the state into the QLearningTable and the QLearningTable chooses an action
  def step(self, obs):
    super(RLAgent, self).step(obs)
    state = str(self.get_state(obs))
    action = self.qtable.choose_action(state)
    if self.previous_action is not None:
      self.qtable.learn(self.previous_state, self.previous_action, obs.reward, 'terminal' if obs.last() else state)
    self.previous_state = state
    self.previous_action = action
    return getattr(self, action)(obs)