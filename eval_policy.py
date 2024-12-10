"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py. I wrote this file to demonstrate
	that our trained policy exists independently of our learning algorithm,
	which resides in ppo.py. Thus, we can test our trained policy without 
	relying on ppo.py.
"""

from torch.distributions import Categorical
from ppo_optimized import greedy

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
    Returns a generator to roll out each episode with early stopping for invalid actions
    """
    MAX_INVALID_ACTIONS = 50  # Threshold for early stopping
    
    for ep in range(1000):
        obs, _ = env.reset(seed=ep)
        done = False
        t = 0
        ep_len = 0
        ep_ret = 0
        invalid_count = 0  # Track invalid actions

        while not done:
            t += 1

            if render:
                env.render()

            # Get stock and product logits from policy
            stock_logits, product_logits = policy(obs)
            
            # Create distributions and sample actions
            stock_dist = Categorical(logits=stock_logits)
            product_dist = Categorical(logits=product_logits)
            stock_action = stock_dist.sample().cpu()
            product_action = product_dist.sample().cpu()

            # Get product size
            products_array = obs['products']
            if product_action.item() < len(products_array):
                products_size = [products_array[product_action.item()]['size'][0], 
                               products_array[product_action.item()]['size'][1]]
            else:
                products_size = [0, 0]

            # Get final action using greedy placement
            action = greedy(obs['stocks'], stock_action.item(), products_size)
            
            # Check for invalid action
            if action['stock_idx'] == -1:
                invalid_count += 1
                if invalid_count >= MAX_INVALID_ACTIONS:
                    print(f"Early stopping due to {invalid_count} invalid actions")
                    done = True
                    ep_ret -= 1.0  # Penalty for too many invalid actions
                    continue

            # Execute action
            obs, rew, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if done:
                print(f"Amanzing! Episode {ep} finished in {t} steps")

            ep_ret += rew

        ep_len = t
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