"""
    The file contains the PPO class to train with.
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
          It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import time
import gymnasium as gym

import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F


# ----------------------------------------------------
# CNN-based Encoder Networks
# ----------------------------------------------------
class StockCNNEncoder(nn.Module):
    def __init__(self):
        super(StockCNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 9 * 9, 128)  # Adjust if exact shape differs

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class ProductEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, embed_dim=32):
        super(ProductEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, prod_features):
        x = F.relu(self.fc1(prod_features))
        x = F.relu(self.fc2(x))
        return x


# ----------------------------------------------------
# Actor Network
# ----------------------------------------------------
class ActorNetwork(nn.Module):
    def __init__(self, num_stocks=100, num_products=20):
        super(ActorNetwork, self).__init__()
        self.num_stocks = num_stocks
        self.num_products = num_products

        self.stock_encoder = StockCNNEncoder()
        self.product_encoder = ProductEncoder()

        # Policy heads
        self.stock_head = nn.Linear(160, num_stocks)
        self.product_head = nn.Linear(160, num_products)

    def forward(self, stock_images, product_features):
        # stock_images: (B, num_stocks, 100, 100)
        # product_features: (B, num_products, 3)

        B = stock_images.size(0)

        # Process stocks
        stock_images = stock_images.unsqueeze(2)  # (B, num_stocks, 1, 100, 100)
        stock_images = stock_images.view(B * self.num_stocks, 1, 100, 100)
        stock_embeds = self.stock_encoder(stock_images)  # (B*num_stocks, 128)
        stock_embeds = stock_embeds.view(B, self.num_stocks, 128)

        # Process products
        product_features = product_features.view(B * self.num_products, 3)
        product_embeds = self.product_encoder(product_features)  # (B*num_products, 32)
        product_embeds = product_embeds.view(B, self.num_products, 32)

        # Aggregate
        stock_summary = stock_embeds.mean(dim=1)     # (B, 128)
        product_summary = product_embeds.mean(dim=1) # (B, 32)
        combined = torch.cat([stock_summary, product_summary], dim=-1)  # (B, 160)

        stock_logits = self.stock_head(combined)       # (B, num_stocks)
        product_logits = self.product_head(combined)   # (B, num_products)

        return stock_logits, product_logits


# ----------------------------------------------------
# Critic Network
# ----------------------------------------------------
class CriticNetwork(nn.Module):
    def __init__(self, num_stocks=100, num_products=20):
        super(CriticNetwork, self).__init__()
        self.num_stocks = num_stocks
        self.num_products = num_products

        self.stock_encoder = StockCNNEncoder()
        self.product_encoder = ProductEncoder()

        # Value head
        self.value_head = nn.Linear(160, 1)

    def forward(self, stock_images, product_features):
        # stock_images: (B, num_stocks, 100, 100)
        # product_features: (B, num_products, 3)

        B = stock_images.size(0)

        # Process stocks
        stock_images = stock_images.unsqueeze(2)  # (B, num_stocks, 1, 100, 100)
        stock_images = stock_images.view(B * self.num_stocks, 1, 100, 100)
        stock_embeds = self.stock_encoder(stock_images)  # (B*num_stocks, 128)
        stock_embeds = stock_embeds.view(B, self.num_stocks, 128)

        # Process products
        product_features = product_features.view(B * self.num_products, 3)
        product_embeds = self.product_encoder(product_features)  # (B*num_products, 32)
        product_embeds = product_embeds.view(B, self.num_products, 32)

        # Aggregate
        stock_summary = stock_embeds.mean(dim=1)     # (B, 128)
        product_summary = product_embeds.mean(dim=1) # (B, 32)
        combined = torch.cat([stock_summary, product_summary], dim=-1)  # (B, 160)

        value = self.value_head(combined)              # (B, 1)

        return value.squeeze(-1)  # (B,)


# ----------------------------------------------------
# PPO Class with Separate Actor and Critic Networks
# ----------------------------------------------------
class PPO:
    """
        This is the PPO class we will use as our model in main.py
    """
    def __init__(self, env, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.

            Parameters:
                env (gym.Env): The environment to train on.
                hyperparameters (dict): All extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """
        
        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env

        # Assuming observations contain 'stocks' and 'products'
        initial_obs, _ = self.env.reset()
        self.num_stocks = len(initial_obs["stocks"])
        self.max_h, self.max_w = initial_obs["stocks"][0].shape
        self.num_products = env.unwrapped.max_product_type

        # Initialize Actor and Critic networks
        self.actor = ActorNetwork(num_stocks=self.num_stocks, num_products=self.num_products).to(self.device)    # ALG STEP 1
        self.critic = CriticNetwork(num_stocks=self.num_stocks, num_products=self.num_products).to(self.device)

        # Initialize separate optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize entropy coefficient
        self.entropy_coef = self.entropy_coef  # Already set in hyperparameters

        # Initialize a dictionary to track stock usage (specific to your environment)
        self.stock_usage = {}

        # Initialize logger for tracking training progress
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
            'critic_losses': [],    # losses of critic network in current iteration
            'lr': self.lr,
        }

    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps (int): The total number of timesteps to train for.

            Return:
                None
        """
        print(f"Learning... {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        initial_lr = self.lr  # Store initial learning rate for annealing

        while t_so_far < total_timesteps:  # ALG STEP 2
            # Collect trajectories by interacting with the environment
            results = self.rollout()  # ALG STEP 3
            batch_obs, batch_stocks, batch_products, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_products_quantities = results

            # Calculate advantages using GAE
            V = self.critic(batch_stocks, batch_products)
            A_k = batch_rews - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Update the number of timesteps and iterations
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Update networks
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Learning rate annealing
                frac = 1.0 - (t_so_far / total_timesteps)
                new_lr = initial_lr * frac
                new_lr = max(new_lr, 1e-7)  # Minimum learning rate

                # Update learning rates for both optimizers
                for param_group in self.actor_optim.param_groups:
                    param_group["lr"] = new_lr
                for param_group in self.critic_optim.param_groups:
                    param_group["lr"] = new_lr
                self.logger['lr'] = new_lr

                # Shuffle indices for mini-batch sampling
                indices = np.arange(batch_obs.size(0))
                np.random.shuffle(indices)
                minibatch_size = batch_obs.size(0) // self.num_minibatches

                for start in range(0, batch_obs.size(0), minibatch_size):
                    end = start + minibatch_size
                    mb_inds = indices[start:end]

                    # Extract mini-batch
                    mb_obs = batch_obs[mb_inds]
                    mb_stocks = batch_stocks[mb_inds]
                    mb_products = batch_products[mb_inds]
                    mb_acts = batch_acts[mb_inds]
                    mb_log_probs = batch_log_probs[mb_inds]
                    mb_advantages = A_k[mb_inds]
                    mb_returns = batch_rews[mb_inds]

                    # Normalize advantages
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-10)

                    # Evaluate current log probabilities and entropy using Actor
                    curr_log_probs, entropy = self.evaluate(mb_stocks, mb_products, mb_acts)

                    # Compute ratios for PPO
                    ratios = torch.exp(curr_log_probs - mb_log_probs)

                    # Compute surrogate losses
                    surr1 = ratios * mb_advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mb_advantages

                    # Compute actor loss
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Compute critic loss
                    V_pred = self.critic(mb_stocks, mb_products)
                    critic_loss = F.mse_loss(V_pred, mb_returns)

                    # Compute entropy loss
                    entropy_loss = -self.entropy_coef * entropy.mean()

                    # Total loss for actor
                    total_actor_loss = actor_loss + entropy_loss

                    # Backward pass and optimize Actor
                    self.actor_optim.zero_grad()
                    total_actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    # Backward pass and optimize Critic
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    # Log losses
                    self.logger['actor_losses'].append(actor_loss.detach().cpu().item())
                    self.logger['critic_losses'].append(critic_loss.detach().cpu().item())

                # Log training summary
                self._log_summary()

                # Save the model at specified frequency
                if i_so_far % self.save_freq == 0:
                    save_path = './ppo_actor_critic.pth'
                    torch.save({
                        'actor_state_dict': self.actor.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                    }, save_path)
                    print(f"Model saved to {save_path}", flush=True)

    def rollout(self):
        """
            Collect trajectories by interacting with the environment.

            Returns:
                Tuple containing:
                    batch_obs (torch.Tensor): Observations collected.
                    batch_stocks (torch.Tensor): Stock images.
                    batch_products (torch.Tensor): Product features.
                    batch_acts (torch.Tensor): Actions taken.
                    batch_log_probs (torch.Tensor): Log probabilities of actions.
                    batch_rews (torch.Tensor): Rewards collected.
                    batch_lens (list): Lengths of episodes.
                    batch_products_quantities (torch.Tensor): Product quantities for masking.
        """
        batch_obs = []
        batch_stocks = []
        batch_products = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_products_quantities = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        while t < self.timesteps_per_batch:
            ep_rews = []
            ep_lens = 0
            obs, _ = self.env.reset()
            done = False

            while not done and ep_lens < self.max_timesteps_per_episode:
                if self.render:
                    self.env.render()

                # Extract observation components
                stocks_np = obs['stocks']  # shape (num_stocks, 100, 100)
                products_np = obs['products']  # shape (num_products, 3)

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
                pad_length = self.num_products - len(products_list)

                # Pad both arrays if needed
                if pad_length > 0:
                    products_list += [[0, 0, 0]] * pad_length
                    products_quantities += [0] * pad_length

                # Convert to numpy arrays
                products_array = np.array(products_list)  # Shape: (num_products, 3)
                products_quantities = np.array(products_quantities)  # Shape: (num_products,)

                # Convert to tensors
                stock_tensor = torch.tensor(np.array(stocks_np), dtype=torch.float).unsqueeze(0).to(self.device)  # (1, num_stocks, 100, 100)
                product_tensor = torch.tensor(products_array, dtype=torch.float).unsqueeze(0).to(self.device)  # (1, num_products, 3)
                product_quant_tensor = torch.tensor(products_quantities, dtype=torch.float).unsqueeze(0).to(self.device)  # (1, num_products)

                # Get action and log probability from Actor
                stock_action, product_action, log_prob = self.get_action(stock_tensor, product_tensor, product_quant_tensor)

                # Store data
                batch_obs.append((stock_tensor, product_tensor))
                batch_stocks.append(stock_tensor)
                batch_products.append(product_tensor)
                batch_acts.append([stock_action, product_action])
                batch_log_probs.append(log_prob)
                batch_products_quantities.append(product_quant_tensor)

                # Prepare the action for the environment
                action = greedy(stocks_np, stock_action, products_np[product_action]['size'])

                # Step the environment
                obs, rew, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Apply usage penalty and rewards
                reward = self.calculate_reward(action, stocks_np, products_np, rew)

                ep_rews.append(reward)
                t += 1
                ep_lens += 1

            batch_lens.append(ep_lens)
            batch_rews.append(ep_rews)

        # Convert lists to tensors
        batch_obs = torch.cat(batch_stocks, dim=0)  # Assuming stock_images and product_features are already tensors
        batch_products = torch.cat(batch_products, dim=0)
        batch_acts = torch.tensor(batch_acts, dtype=torch.long).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)
        batch_rews = self.compute_rtgs(batch_rews).to(self.device)
        batch_products_quantities = torch.cat(batch_products_quantities, dim=0)

        return batch_obs, batch_products, batch_products, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_products_quantities

    def get_action(self, stock_tensor, product_tensor, product_quant_tensor):
        """
            Queries an action from the actor network.

            Parameters:
                stock_tensor (torch.Tensor): Stock images tensor.
                product_tensor (torch.Tensor): Product features tensor.
                product_quant_tensor (torch.Tensor): Product quantities tensor.

            Return:
                stock_action (int): Selected stock index.
                product_action (int): Selected product index.
                log_prob (float): Log probability of the selected action.
        """
        # Forward pass through Actor network
        stock_logits, product_logits = self.actor(stock_tensor, product_tensor)

        # Apply temperature annealing if needed (optional)
        # temperature = max(1.0 - (self.logger['t_so_far'] / self.timesteps_per_batch), 0.1)
        # stock_logits = stock_logits / temperature
        # product_logits = product_logits / temperature

        # Apply mask to product logits based on available quantities
        product_mask = (product_quant_tensor > 0).squeeze(0)  # Shape: (num_products,)
        product_logits = product_logits.masked_fill(~product_mask, -float('inf'))

        # Create distributions
        stock_dist = Categorical(logits=stock_logits)
        product_dist = Categorical(logits=product_logits)

        # Sample actions
        stock_action = stock_dist.sample()
        product_action = product_dist.sample()

        # Compute log probabilities
        log_prob = stock_dist.log_prob(stock_action) + product_dist.log_prob(product_action)

        return stock_action.item(), product_action.item(), log_prob.item()

    def evaluate(self, batch_stocks, batch_products, batch_acts):
        """
            Estimate the values of each observation and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_stocks (torch.Tensor): Stock images batch.
                batch_products (torch.Tensor): Product features batch.
                batch_acts (torch.Tensor): Actions batch.

            Return:
                V (torch.Tensor): Value estimates.
                log_probs (torch.Tensor): Log probabilities of actions.
                entropy (torch.Tensor): Entropy of the action distributions.
        """
        # Forward pass through Actor network
        stock_logits, product_logits = self.actor(batch_stocks, batch_products)

        # Apply mask to product logits
        product_quantities = (batch_products_quantities > 0)  # Shape: (batch_size, num_products)
        product_logits = product_logits.masked_fill(~product_quantities, -float('inf'))

        # Create distributions
        stock_dist = Categorical(logits=stock_logits)
        product_dist = Categorical(logits=product_logits)

        # Gather actions
        stock_actions = torch.tensor([act[0] for act in batch_acts]).to(self.device)
        product_actions = torch.tensor([act[1] for act in batch_acts]).to(self.device)

        # Calculate log probabilities
        stock_log_probs = stock_dist.log_prob(stock_actions)
        product_log_probs = product_dist.log_prob(product_actions)
        log_probs = stock_log_probs + product_log_probs

        # Calculate entropy
        entropy = stock_dist.entropy() + product_dist.entropy()

        # Forward pass through Critic network for value estimates
        V = self.critic(batch_stocks, batch_products)

        return V.squeeze(), log_probs, entropy

    def compute_rtgs(self, batch_rews):
        """
            Compute Rewards-To-Go (RTG) for each timestep.

            Parameters:
                batch_rews (list of list of float): Rewards collected per episode.

            Return:
                torch.Tensor: Rewards-To-Go for each timestep.
        """
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + self.gamma * discounted_reward
                batch_rtgs.insert(0, discounted_reward)
        return torch.tensor(batch_rtgs, dtype=torch.float)

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters (dict): The extra arguments included when creating the PPO model, should only include
                                        hyperparameters defined below with custom values.

            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate
        self.gamma = 0.95                               # Discount factor for Rewards-To-Go
        self.clip = 0.2                                 # Clipping parameter for PPO
        self.lam = 0.95                                 # Lambda for GAE
        self.num_minibatches = 4                        # Number of mini-batches for updates
        self.entropy_coef = 0.01                        # Entropy coefficient for exploration
        self.target_kl = 0.02                           # KL Divergence threshold
        self.max_grad_norm = 0.5                        # Gradient clipping threshold

        # Miscellaneous parameters
        self.render = False                             # If we should render during rollout
        self.save_freq = 10                             # How often to save in number of iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device (CPU or GPU)
        self.seed = None								# Sets the seed of our program, used for reproducibility of results

        # Update hyperparameters with any custom values
        for param, val in hyperparameters.items():
            setattr(self, param, val)
        
        # Set the seed if specified
        if self.seed is not None:
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        lr = self.logger['lr']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean(self.logger['actor_losses']) if self.logger['actor_losses'] else 0
        avg_critic_loss = np.mean(self.logger['critic_losses']) if self.logger['critic_losses'] else 0

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        avg_critic_loss = str(round(avg_critic_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
        print(f"Average Critic Loss: {avg_critic_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Learning rate: {lr}", flush=True)
        print("------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []

    def calculate_reward(self, action, stocks_np, products_np, original_rew):
        """
            Calculate the reward based on the action and environment state.

            Parameters:
                action (dict): The action taken.
                stocks_np (list): Current stock states.
                products_np (list): Current product states.
                original_rew (float): Original reward from the environment.

            Return:
                reward (float): Calculated reward.
        """
        reward = original_rew

        stock_idx = action['stock_idx']
        product_idx = action['product_idx']

        if stock_idx not in self.stock_usage:
            self.stock_usage[stock_idx] = 0
        self.stock_usage[stock_idx] += 1

        # Apply usage penalty
        usage_penalty = self.stock_usage[stock_idx] * 5
        reward -= usage_penalty

        # Check if a new stock was used
        if self.is_new_stock(stocks_np[stock_idx]):
            reward += 50

        # Check for invalid actions
        if stock_idx == -1 or products_np[product_idx]['quantity'] == 0:
            reward -= 100
        else:
            product_area = products_np[product_idx]['size'][0] * products_np[product_idx]['size'][1]
            stock_area = self._get_stock_area(stocks_np[stock_idx])
            efficiency_bonus = 20 * (product_area / stock_area) if stock_area > 0 else 0
            reward += 10 + efficiency_bonus

        return reward

    def is_new_stock(self, stock):
        """
            Determine if a stock is new based on its state.

            Parameters:
                stock (np.array): Stock state.

            Return:
                bool: True if new stock, False otherwise.
        """
        return np.all(stock < 0)

    def _get_stock_area(self, stock):
        """
            Calculate the area of a stock.

            Parameters:
                stock (np.array): Stock state.

            Return:
                int: Area of the stock.
        """
        w, h = _get_stock_size_(stock)
        return w * h


def greedy(stocks, stock_idx, prod_size):
    """
        Greedy placement strategy for products on stocks.

        Parameters:
            stocks (list): List of stock states.
            stock_idx (int): Index of the selected stock.
            prod_size (list): Size of the product.

        Return:
            dict: Action dictionary containing stock index, size, and position.
    """
    stock = stocks[stock_idx]
    stock_w, stock_h = _get_stock_size_(stock)
    prod_w, prod_h = prod_size

    # Try placing the product in original orientation
    if stock_w >= prod_w and stock_h >= prod_h:
        for x in range(stock_w - prod_w + 1):
            find = 0
            for y in range(stock_h - prod_h + 1):
                if stock[x][y] == -1:
                    if find == 0:
                        if _can_place_(stock, (x, y), prod_size):
                            return {"stock_idx": stock_idx, "product_idx": -1, "size": prod_size, "position": (x, y)}
                        find = 1
                else:
                    if find == 1:
                        find = 0

    # Try placing the product in rotated orientation
    if stock_w >= prod_h and stock_h >= prod_w:
        for x in range(stock_w - prod_h + 1):
            find = 0
            for y in range(stock_h - prod_w + 1):
                if stock[x][y] == -1:
                    if find == 0:
                        if _can_place_(stock, (x, y), prod_size[::-1]):
                            return {"stock_idx": stock_idx, "product_idx": -1, "size": prod_size[::-1], "position": (x, y)}
                        find = 1
                else:
                    if find == 1:
                        find = 0

    # If placement fails, return invalid action
    return {"stock_idx": -1, "product_idx": -1, "size": [0, 0], "position": (0, 0)}


def _get_stock_size_(stock):
    """
        Calculate the size of a stock.

        Parameters:
            stock (np.array): Stock state.

        Return:
            tuple: Width and height of the stock.
    """
    stock_w = np.sum(np.any(stock != -2, axis=1))
    stock_h = np.sum(np.any(stock != -2, axis=0))

    return stock_w, stock_h


def _can_place_(stock, position, prod_size):
    """
        Check if a product can be placed at a given position on the stock.

        Parameters:
            stock (np.array): Stock state.
            position (tuple): (x, y) position to place the product.
            prod_size (list): [width, height] of the product.

        Return:
            bool: True if placement is possible, False otherwise.
    """
    pos_x, pos_y = position
    prod_w, prod_h = prod_size

    return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)
