from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import features
from absl import app
import ZAgent.Agent as Z
import gui as UI
import RLAgent.RLAgent as RL
import RLAgent.RandomAgent as RA
from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop
import threading




# def init_agents():
#     zAgent = Z.ZergAgent()
    
#     try:
#         while True:
#             with sc2_env.SC2Env(
#                     map_name="AbyssalReef",
#                     players=[sc2_env.Agent(sc2_env.Race.zerg), 
#                              sc2_env.Bot(sc2_env.Race.random,sc2_env.Difficulty.very_hard)],
#                     agent_interface_format=features.AgentInterfaceFormat(
#                         feature_dimensions=features.Dimensions(screen=84, minimap=64),
#                         use_feature_units=True),
#                     step_mul=16,
#                     game_steps_per_episode=0,
#                     visualize=True) as env:

#                 #Setup Agents
#                 zAgent.setup(env.observation_spec(), env.action_spec())
              

#                 timesteps = env.reset()
#                 zAgent.reset()
       

#                 while True:
#                     step_actions = [zAgent.step(timesteps[0])]
#                     if timesteps[0].last():
#                         break
#                     timesteps = env.step(step_actions)
                    
            
#     except KeyboardInterrupt:
#         pass

def init_agents():
  agent1 = RL.RLAgent()
  agent2 = RA.RandomAgent()
  try:
    with sc2_env.SC2Env(
        map_name="Simple64", # Choose the map
        players=[sc2_env.Agent(sc2_env.Race.terran), 
                 sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=features.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            feature_dimensions=features.Dimensions(screen=84, minimap=64),
            use_feature_units=True,
            raw_resolution=64,
        ),
        step_mul=128, # How fast it runs the game
        disable_fog=True, # Too see everything in the minimap
    ) as env:
      run_loop.run_loop([agent1, agent2], env, max_episodes=1000) # Control both agents instead of one
  except KeyboardInterrupt:
    env.close()
    pass


def main(unused_argv):
  t1 = threading.Thread(target=init_agents)
  t1.start()
  ui = UI.GUI("RL Agent Starcraft II", "1800x900")

  ui.print_to_tab("Running tests ....", ui.console)

  
  ui.window.mainloop()
  t1.join()


if __name__ == "__main__":
    app.run(main)
