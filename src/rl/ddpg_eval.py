import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer

from env_moveit import Env

class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.feature_extractor = nn.Sequential(nn.Conv3d(1, 8, kernel_size=5, stride=2, padding=2),
                                               nn.ReLU(),
                                               nn.MaxPool3d(4,stride=2),
                                               nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
                                               nn.ReLU(),
                                               nn.MaxPool3d(4,stride=2),
                                               nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
                                               nn.ReLU(),
                                               nn.Flatten())
        
        self.net = nn.Sequential(nn.Linear(46,256),
                                 nn.ReLU(),
                                 nn.Linear(256,128),
                                 nn.ReLU(),
                                 nn.Linear(128,7),
                                 nn.Tanh())

    def compute(self, inputs, role):
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)

        voxel = space["s3d"]
        pose = space["pose"]

        feature = self.feature_extractor(voxel)
        feature_pose = torch.cat([feature,pose],dim=1)
        return self.net(feature_pose), {}
    
class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.feature_extractor = nn.Sequential(nn.Conv3d(1, 8, kernel_size=5, stride=2, padding=2),
                                               nn.ReLU(),
                                               nn.MaxPool3d(4,stride=2),
                                               nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
                                               nn.ReLU(),
                                               nn.MaxPool3d(4,stride=2),
                                               nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
                                               nn.ReLU(),
                                               nn.Flatten())
        
        self.net = nn.Sequential(nn.Linear(53,128),
                                 nn.ReLU(),
                                 nn.Linear(128,64),
                                 nn.ReLU(),
                                 nn.Linear(64,1))

    def compute(self, inputs, role):
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)

        voxel = space["s3d"]
        pose = space["pose"]

        feature = self.feature_extractor(voxel)
        feature_pose = torch.cat([feature,pose],dim=1)
        return self.net(torch.cat([feature_pose, inputs["taken_actions"]], dim=1)), {}
    

env = Env()
env = wrap_env(env)

device = env.device

models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device)
models["target_policy"] = Actor(env.observation_space, env.action_space, device)
models["critic"] = Critic(env.observation_space, env.action_space, device)
models["target_critic"] = Critic(env.observation_space, env.action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#configuration-and-hyperparameters
cfg = DDPG_DEFAULT_CONFIG.copy()
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 10
cfg["experiment"]["checkpoint_interval"] = 0
cfg["experiment"]["directory"] = "runs/eval"
cfg["experiment"]["experiment_name"] = "2.0"

agent = DDPG(models=models,
             memory=None,
             cfg=cfg,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
# trainer.train()

agent.load("runs/DDPG/1.5/checkpoints/best_agent.pt")

# start evaluation
trainer.eval()