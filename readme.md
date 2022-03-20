#Usage
#Hardware Requirements
- for fast training, a GPU is recommended. We ran each experiment on a single Titan X Pascal (12GB).

#Software Requirements
- Python 3
- pysc2 (tested with v1.2)
- TensorFlow (tested with 1.4.0)
- StarCraft II and mini games (see below or pysc2)

#Quick Install Guide
- pip install numpy tensorflow-gpu pysc2==1.2
- Install StarCraft II. On Linux, use 3.16.1.
- Download the mini games and extract them to your StarcraftII/Maps/ directory.

#Train & run
- run and train: python3 main.py 01234 --train --nhwc --map AbyssalReef.
- run and evalutate without training: python3 main.py 01234 --nhwc --map AbyssalReef.

You can visualize the agents during training or evaluation with the --vis flag. See run.py for all arguments.

Summaries are written to out/summary/<experiment_name> and model checkpoints are written to out/models/<experiment_name>.

#Acknowledgments
- The code in Networks/environment.py is based on OpenAI baselines, with adaptions from sc2aibot. 
- Some of the code in RLAgent/runner.py is loosely based on sc2aibot.