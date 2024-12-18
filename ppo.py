import gymnasium as gym
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical

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
    def __init__(self, env, **hyperparameters):
        # Check GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self._init_hyperparameters(hyperparameters)
        observation, _ = env.reset()
        self.env = env

        self.num_stocks = len(observation["stocks"])
        self.max_h, self.max_w = observation["stocks"][0].shape
        self.num_products = env.unwrapped.max_product_type

        # Initialize Actor and Critic networks
        self.actor = ActorNetwork(num_stocks=self.num_stocks, num_products=self.num_products).to(self.device)
        self.critic = CriticNetwork(num_stocks=self.num_stocks, num_products=self.num_products).to(self.device)

        # Initialize separate optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Logger
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lens': [],
            'batch_rews': [],
            'actor_losses': [],
            'critic_losses': [],
        }
        
        self.stock_usage = {}  # Track stock usage


    def learn(self, total_timesteps):
        print(f"Learning... {self.max_timesteps_per_episode} steps/ep, {self.timesteps_per_batch} steps/batch, total {total_timesteps} steps")
        t_so_far = 0
        i_so_far = 0
        # Store initial learning rate
        initial_lr = self.lr
        while t_so_far < total_timesteps:
            results = self.rollout()
            batch_obs, batch_stocks, batch_products, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_products_quantities = results

            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Evaluate old actions and values
            V = self.critic(batch_stocks, batch_products)
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Update networks
            for _ in range(self.n_updates_per_iteration):
                # Learning rate annealing
                frac = 1.0 - (t_so_far / total_timesteps)
                new_lr = initial_lr * frac
                new_lr = max(new_lr, 1e-7)  # Minimum learning rate
                
                # Update learning rates
                for param_group in self.actor_optim.param_groups:
                    param_group["lr"] = new_lr
                for param_group in self.critic_optim.param_groups:
                    param_group["lr"] = new_lr
                
                # Zero gradients
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()

                # Forward pass through Actor
                stock_logits, product_logits = self.actor(batch_stocks, batch_products)

                # Apply temperature annealing
                temperature = max(1.0 - (self.logger['t_so_far'] / self.timesteps_per_batch), 0.1)
                stock_logits = stock_logits / temperature
                product_logits = product_logits / temperature

                # Apply mask to product logits
                product_mask = (batch_products_quantities > 0)  # Shape: (batch_size, num_products)
                product_logits = product_logits.masked_fill(~product_mask, -float('inf'))

                # Create distributions
                stock_dist = Categorical(logits=stock_logits)
                product_dist = Categorical(logits=product_logits)

                # Get log probabilities of the taken actions
                stock_actions = batch_acts[:, 0]
                product_actions = batch_acts[:, 1]
                stock_log_probs = stock_dist.log_prob(stock_actions)
                product_log_probs = product_dist.log_prob(product_actions)
                log_probs = stock_log_probs + product_log_probs

                # Ratios for PPO
                ratios = torch.exp(log_probs - batch_log_probs)

                # Surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy for exploration
                entropy = stock_dist.entropy() + product_dist.entropy()
                entropy_loss = -self.entropy_coef * entropy.mean()

                # Total actor loss
                total_actor_loss = actor_loss + entropy_loss

                # Backward pass for Actor
                total_actor_loss.backward()
                self.actor_optim.step()

                # Forward pass through Critic
                values = self.critic(batch_stocks, batch_products)
                critic_loss = F.mse_loss(values, batch_rtgs)

                # Backward pass for Critic
                critic_loss.backward()
                self.critic_optim.step()

                # Log the losses
                self.logger['actor_losses'].append(actor_loss.detach().cpu().item())
                self.logger['critic_losses'].append(critic_loss.detach().cpu().item())

            self._log_summary()

            if i_so_far % self.save_freq == 0:
                torch.save({
                    'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                }, './ppo_actor_critic.pth')

    def rollout(self):
        self.stock_usage = {}

        batch_obs = []
        batch_stocks = []
        batch_products = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        batch_products_quantities = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs, _ = self.env.reset(seed=t)
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render()

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
                pad_length = self.num_products - len(products_list)

                # Pad both arrays if needed
                if pad_length > 0:
                    products_list += [[0, 0, 0]] * pad_length
                    products_quantities += [0] * pad_length

                # Convert to numpy arrays
                products_array = np.array(products_list)  # Shape: (num_products, 3)
                products_quantities = np.array(products_quantities)  # Shape: (num_products,)

                # Convert to tensors
                stocks_tensor = torch.tensor(np.array(stocks_np), dtype=torch.float).unsqueeze(0).to(self.device)
                products_tensor = torch.tensor(products_array, dtype=torch.float).unsqueeze(0).to(self.device)
                products_quantities_tensor = torch.tensor(products_quantities, dtype=torch.float).unsqueeze(0).to(self.device)
                
                # Get action with masking
                stock_action, product_action, log_prob = self.get_action(stocks_tensor, products_tensor, products_quantities_tensor)


                # Store data
                batch_obs.append(obs)
                batch_stocks.append(stocks_np)
                batch_products.append(products_array)
                batch_acts.append([stock_action, product_action])
                batch_log_probs.append(log_prob)
                batch_products_quantities.append(products_quantities) # Store quantities for masking


                action = greedy(stocks_np, stock_action, products_np[product_action]['size'])
                reward = 0
                
                if stock_action not in self.stock_usage:
                    self.stock_usage[stock_action] = 0
                self.stock_usage[stock_action] += 1
                
                # Apply usage penalty
                usage_penalty = self.stock_usage[stock_action] * 5
                reward -= usage_penalty
                if self.is_new_stock(obs['stocks'][stock_action]):
                    reward += 50
                
                if action['stock_idx'] == -1 or products_np[product_action]['quantity'] == 0:
                    reward -= 100
                else:
                    product_area = products_np[product_action]['size'][0] * products_np[product_action]['size'][1]
                    stock_area = self._get_stock_area(obs['stocks'][stock_action])
                    efficiency_bonus = 20 * (product_area / stock_area) if stock_area > 0 else 0
                    reward += 10 + efficiency_bonus
                
                obs, rew, terminated, truncated, info = self.env.step(action)
                
                done = terminated or truncated
                if done:
                    if not self.fullfill(obs['products']):
                        unfilled, total_products = 0, 0
                        for product in obs['products']:
                            total_products += 1
                            if product['quantity'] > 0:
                                unfilled += 1
                        reward -= 100 * (total_products - unfilled)
                    reward -= 100 * info['trim_loss']
                ep_rews.append(reward)
                t += 1

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Convert to tensors

        batch_rtgs = self.compute_rtgs(batch_rews)
        
        batch_stocks = torch.tensor(np.array(batch_stocks), dtype=torch.float).to(self.device)
        batch_products = torch.tensor(np.array(batch_products), dtype=torch.float).to(self.device) 
        batch_acts = torch.tensor(batch_acts, dtype=torch.long).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)
        if isinstance(batch_rtgs, torch.Tensor):
            batch_rtgs = batch_rtgs.clone().detach().float().to(self.device)
        else:
            batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)
            
        batch_products_quantities = torch.tensor(np.array(batch_products_quantities), dtype=torch.float).to(self.device)



        # Logging
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_stocks, batch_products, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_products_quantities
    
    
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + self.gamma * discounted_reward
                batch_rtgs.insert(0, discounted_reward)
        return torch.tensor(batch_rtgs, dtype=torch.float)

    def get_action(self, stocks_tensor, products_tensor, products_quantities):
        # Forward pass through Actor network
        stock_logits, product_logits = self.actor(stocks_tensor, products_tensor)
        
        # Apply temperature annealing
        temperature = max(1.0 - (self.logger['t_so_far'] / self.timesteps_per_batch), 0.1)
        stock_logits = stock_logits / temperature
        product_logits = product_logits / temperature
        
        # Apply mask to product logits
        product_mask = (products_quantities > 0).squeeze(0)  # Shape: (num_products,)
        # Set logits of unavailable products to -inf
        product_logits = product_logits.masked_fill(~product_mask, -float('inf'))


        # Create distributions
        stock_dist = Categorical(logits=stock_logits)
        product_dist = Categorical(logits=product_logits)

        stock_action = stock_dist.sample()
        product_action = product_dist.sample()

        log_prob = stock_dist.log_prob(stock_action) + product_dist.log_prob(product_action)

        return stock_action.item(), product_action.item(), log_prob.item()

    def evaluate(self, batch_stocks, batch_products, batch_acts, batch_products_quantities):
        # Forward pass through Actor network
        stock_logits, product_logits = self.actor(batch_stocks, batch_products)

        # Apply mask to product logits
        product_mask = (batch_products_quantities > 0)  # Shape: (batch_size, num_products)
        product_logits = product_logits.masked_fill(~product_mask, -float('inf'))

        stock_dist = Categorical(logits=stock_logits)
        product_dist = Categorical(logits=product_logits)

        stock_actions = batch_acts[:, 0]
        product_actions = batch_acts[:, 1]

        stock_log_probs = stock_dist.log_prob(stock_actions)
        product_log_probs = product_dist.log_prob(product_actions)
        
        stock_entropy = stock_dist.entropy()
        product_entropy = product_dist.entropy()
        entropy = stock_entropy + product_entropy

        log_probs = stock_log_probs + product_log_probs

        # Forward pass through Critic network
        values = self.critic(batch_stocks, batch_products)

        return values.squeeze(), log_probs, entropy

    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 5
        self.lr = 0.005
        self.gamma = 0.95
        self.clip = 0.2

        self.render = True
        self.render_every_i = 10
        self.save_freq = 10
        self.seed = None

        for param, val in hyperparameters.items():
            setattr(self, param, val)

        if self.seed is not None:
            torch.manual_seed(self.seed)
            print(f"Seed set to {self.seed}")
            
        self.entropy_coef = 0.05  # Entropy coefficient for exploration

    def _log_summary(self):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean(self.logger['actor_losses']) if self.logger['actor_losses'] else 0
        avg_critic_loss = np.mean(self.logger['critic_losses']) if self.logger['critic_losses'] else 0

        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        avg_critic_loss = str(round(avg_critic_loss, 5))

        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
        print(f"Average Critic Loss: {avg_critic_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print("------------------------------------------------------", flush=True)
        print(flush=True)

        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []
        
    def is_new_stock(self, stock):
        return np.all(stock < 0)
    
    def fullfill(self, products):
        for product in products:
            if product['quantity'] > 0:
                return False
        return True
    
    def _get_stock_area(self, stock):
        w, h = _get_stock_size_(stock)
        return w * h

def greedy(stocks, stock_idx, prod_size):
    # TODO
    stock = stocks[stock_idx]
    stock_w, stock_h = _get_stock_size_(stock)
    prod_w, prod_h = prod_size

    if stock_w >= prod_w and stock_h >= prod_h:
        for x in range(stock_w - prod_w + 1):
            find = 0
            for y in range(stock_h - prod_h + 1):
                if stock[x][y] == -1:
                    if find == 0:
                        if _can_place_(stock, (x, y), prod_size):
                            return {"stock_idx": stock_idx, "size": prod_size, "position": (x, y)}
                        find = 1
                else:
                    if find == 1:
                        find = 0

    if stock_w >= prod_h and stock_h >= prod_w:
        for x in range(stock_w - prod_h + 1):
            find = 0
            for y in range(stock_h - prod_w + 1):
                if stock[x][y] == -1:
                    if find == 0:
                        if _can_place_(stock, (x, y), prod_size[::-1]):
                            return {"stock_idx": stock_idx, "size": prod_size[::-1], "position": (x, y)}
                        find = 1
                else:
                    if find == 1:
                        find = 0

    return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

def _get_stock_size_(stock):
    stock_w = np.sum(np.any(stock != -2, axis=1))
    stock_h = np.sum(np.any(stock != -2, axis=0))

    return stock_w, stock_h

def _can_place_(stock, position, prod_size):
    pos_x, pos_y = position
    prod_w, prod_h = prod_size

    return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)
