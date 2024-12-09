"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py. I wrote this file to demonstrate
	that our trained policy exists independently of our learning algorithm,
	which resides in ppo.py. Thus, we can test our trained policy without 
	relying on ppo.py.
"""

import torch
import numpy as np
from torch.distributions import Categorical
from ppo import greedy

def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def rollout(policy, env, render):
	"""
		Returns a generator to roll out each episode given a trained policy and
		environment to test on. 

		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
			render - Specifies whether to render or not
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.

		Note:
			If you're unfamiliar with Python generators, check this out:
				https://wiki.python.org/moin/Generators
			If you're unfamiliar with Python "yield", check this out:
				https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	"""
	# Determine the device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	policy = policy.to(device)
	
	num_products = env.unwrapped.max_product_type
    
	# Rollout until user kills process
	while True:
		obs, _ = env.reset()
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return

		while not done:
			t += 1

			# Render environment if specified, off by default
			if render:
				env.render()
    
			# Extract observation components
			stocks_np = obs['stocks']  # shape (num_stocks, 100, 100)
			products_np = obs['products']  # shape (num_products, 3)
			# Extract numerical data from products_np
			# Extract product features and quantities
			products_list = []
			products_quantities = []
			for product in products_np:
				size = product['size']
				quantity = product['quantity']
				product_features = np.concatenate((size, [quantity]))
				products_list.append(product_features)
				products_quantities.append(quantity)

			# Calculate padding length
			pad_length = num_products - len(products_list)

			# Pad both arrays if needed
			if pad_length > 0:
				products_list += [[0, 0, 0]] * pad_length
				products_quantities += [0] * pad_length

			# Convert to numpy arrays
			products_array = np.array(products_list)  # Shape: (num_products, 3)
			products_quantities = np.array(products_quantities)  # Shape: (num_products,)

			# Convert to tensors
			stocks_tensor = torch.tensor(np.array(stocks_np), dtype=torch.float).unsqueeze(0).to(device)
			products_tensor = torch.tensor(products_array, dtype=torch.float).unsqueeze(0).to(device)
			products_quantities_tensor = torch.tensor(products_quantities, dtype=torch.float).unsqueeze(0).to(device)
			
			# Get action with masking
			# stock_action, product_action, log_prob = model.get_action(stocks_tensor, products_tensor, products_quantities_tensor)

			# Query deterministic action from policy and run it
			with torch.no_grad():
				stock_logits, product_logits, _ = policy(stocks_tensor, products_tensor)

				# Apply mask to product logits
				product_mask = (products_quantities_tensor > 0).squeeze(0)  # Shape: (num_products,)
				# Set logits of unavailable products to -inf
				product_logits = product_logits.masked_fill(~product_mask, -float('inf'))

                # Create distributions
				stock_dist = Categorical(logits=stock_logits)
				product_dist = Categorical(logits=product_logits)

                # Sample actions
				stock_action = stock_dist.sample().item()
				product_action = product_dist.sample().item()

            # **Prepare the action for the environment**
			action = greedy(stocks_np, stock_action, products_np[product_action]['size'])

            # **Step the environment**
			obs, rew, terminated, truncated, _ = env.step(action)
			done = terminated or truncated


			# Sum all episodic rewards as we go along
			ep_ret += rew
			
		# Track episodic length
		ep_len = t

		# returns episodic length and return in this iteration
		yield ep_len, ep_ret

def eval_policy(policy, env, render=False):
	"""
		The main function to evaluate our policy with. It will iterate a generator object
		"rollout", which will simulate each episode and return the most recent episode's
		length and return. We can then log it right after. And yes, eval_policy will run
		forever until you kill the process. 

		Parameters:
			policy - The trained policy to test, basically another name for our actor model
			env - The environment to test the policy on
			render - Whether we should render our episodes. False by default.

		Return:
			None

		NOTE: To learn more about generators, look at rollout's function description
	"""
	# Rollout with the policy and environment, and log each episode's data
	for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)