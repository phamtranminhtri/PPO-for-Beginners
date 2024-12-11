"""
	This file is the executable for running PPO. It is based on this medium article: 
	https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gymnasium as gym
import sys
import torch

from arguments import get_args
from ppo_optimized import PPO, ActorNetwork, CriticNetwork
from eval_policy import eval_policy

import gym_cutting_stock

def train(env, hyperparameters, actor_model, critic_model):
    print(f"Training", flush=True)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    # Create a model for PPO with device
    model = PPO(env=env, **hyperparameters)

    # Try loading existing models with proper device mapping
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(
            torch.load(actor_model, map_location=device)
        )
        model.critic.load_state_dict(
            torch.load(critic_model, map_location=device)
        )
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '':
        print(f"Error: Either specify both actor/critic models or none at all.")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model
    model.learn(total_timesteps=200_000_000)

def test(env, actor_model):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)
  
	# Determine device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# Extract out dimensions of observation and action spaces
	observation, _ = env.reset()
	num_stocks = len(observation["stocks"])
	max_h, max_w = observation["stocks"][0].shape
	num_products = env.unwrapped.max_product_type

	# Build our policy the same way we build our actor model in PPO
	policy = ActorNetwork(num_stocks=num_stocks, num_products=num_products).to(device)

	# Load model with appropriate device mapping
	policy.load_state_dict(
        torch.load(
            actor_model,
            map_location=device
        )
    )

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
	eval_policy(policy=policy, env=env, render=True)

def main(args):
	"""
		The main function to run.

		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
	# NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
	# ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
	# To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
	hyperparameters = {
				'timesteps_per_batch': 300,#2048, 
				'max_timesteps_per_episode': 60, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': True,
				'render_every_i': 10
			  }

	# Creates the environment we'll be running. If you want to replace with your own
	# custom environment, note that it must inherit Gym and have both continuous
	# observation and action spaces.# Create the environment
	env = gym.make(
        "gym_cutting_stock/CuttingStock-v0", 
        render_mode='human' if args.mode == 'test' else 'rgb_array',
        num_stocks=16,
        max_product_type=5,
        max_product_per_type=10,
    )

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
	else:
		test(env=env, actor_model=args.actor_model)

if __name__ == '__main__':
	args = get_args() # Parse arguments from command line
	main(args)
