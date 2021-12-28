import random
import RLAgent.Agent as Agent

# Takes the list of actions of the base agent, chooses one at random and then execute it
class RandomAgent(Agent.Agent):
  def step(self, obs):
    super(RandomAgent, self).step(obs)
    action = random.choice(self.actions)
    return getattr(self, action)(obs)