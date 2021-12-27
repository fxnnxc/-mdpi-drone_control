import ray 
from ray import tune 
from ray.tune.registry import register_env 
import numpy as np 
import argparse 
import json 
from drone.train.wind_correction.rl_correction.env import RLCorrectionEnv

first_env = RLCorrectionEnv({"predict_model_path":"../../predict_model/predictor.pt"})

def train(Environment_Class, rllib_config, env_config, checkpoint=None, multiagent_config=None):
    ray.init()
    config = {
        "env": Environment_Class,
        "num_workers":rllib_config['num_workers'],
        "num_gpus": rllib_config['num_gpus'],
        "env_config": env_config,
        "framework":"torch"
    }
    if multiagent_config:
        config["multiagent"]=multiagent_config
    # config['exploration_config'] = {
    #             # The Exploration class to use.
    #             "type": "EpsilonGreedy",
    #             # Config for the Exploration class' constructor:
    #             "initial_epsilon": 1.0,
    #             "final_epsilon": 0.02,
    #             "epsilon_timesteps": 1000,  # Timesteps over which to anneal epsilon.
    #         }
    config["train_batch_size"] = 256
    config['horizon'] = 100
    config['timesteps_per_iteration'] = 256
    config.update({#"learning_starts": 100,
                    "train_batch_size": 256,
                    "rollout_fragment_length": 50,
                    #"target_network_update_freq": 500,
                    #"timesteps_per_iteration": 100, 
                    #"v_min": -100.0,
                   # "v_max": 100.0,
                    })
    # config['sgd_minibatch_size'] = 32
    # config["num_sgd_iter"] = 16

    analysis = tune.run(
        rllib_config['model'],
        config=config,
        stop = rllib_config['stop'],
        checkpoint_freq = rllib_config['checkpoint_freq'],
        checkpoint_at_end=True,
        local_dir = rllib_config['local_dir'],
        name = rllib_config['name'],
        restore= checkpoint
    )

    print("--------------------")
    print("Train is finished!")
    print("--------------------")
    
def test(Environment_Class, rllib_config, env_config, checkpoint, multiagent_config=None):
    ray.init()
    config = {
        "env": Environment_Class,
        "num_workers":0,
        "num_gpus": rllib_config['num_gpus'],
        "env_config": env_config,
        "framework":"torch"
    }
    if multiagent_config:
        config["multiagent"]=multiagent_config

    if rllib_config['model'] == "PPO":
        from ray.rllib.agents.ppo import PPOTrainer
        agent = PPOTrainer(config=config, env=Environment_Class)
    else:
        ValueError()
    
    agent.restore(checkpoint)
    env = Environment_Class(env_config)
    Reward = [] 
    for i in range(100):
        obs = env.reset()
        done = False
        Reward.append(0)
        while not done:
            alive_agents = [i for i in range(env_config['num_agents'])]
            actions =  {i:agent.compute_action(obs[i], policy_id=f"pol_{i}")  for i in alive_agents}
            obs, reward, done, info = env.step(actions)
            done = done["__all__"]
            Reward[-1] += sum(reward.values())/len(reward.values())
            print(i,  Reward[-1])

    print("--------------------")
    print("Test is finished!")
    print("--------------------")



def construct_multiagent_config():
    from gym.spaces import Box, Discrete
    {
        "policies":{f"pol_{i}" : (None, first_env.observation_space, first_env.action_space , {}) for i in range(1)}, # obs, act
        "policy_mapping_fn": lambda i : f"pol_{i}",
        "policies_to_train":[f"pol_{i}" for i in range(env_config.get("num_drones"))],
        "observation_fn" : None # preprocess observation for each agent
    },

def observation_fn(agent_obs):
    new_obs = {}
    for i in agent_obs.keys():
        new_obs[i] = agent_obs[i].flatten()
    return new_obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.json')
    parser.add_argument('--test', action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f :
        config = json.load(f)
        rllib_config = config['rllib_config']
        env_config = config['env_config']
        multi_agent_config  = construct_multiagent_config() #None  # set None if it is not multiagent setting
    print("------")
    env_config['wind_field'] = lambda pos : np.array([1,1,0])/3
    env_config['trajectory'] = [[np.cos(i/20 * np.pi*2), np.sin(i/20 * np.pi*2), 1+i/20] for i in range(60)]
    print(env_config)
    print("------")
    Environment_Class = RLCorrectionEnv

    if args.test:
        test(Environment_Class, rllib_config, env_config, args.checkpoint, multi_agent_config)
    else:
        train(Environment_Class, rllib_config, env_config, args.checkpoint, multi_agent_config)