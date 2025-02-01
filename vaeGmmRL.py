import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import gym
from gym import spaces

# =========================== 1. VAE Definition ============================= #
class VAE(nn.Module):
    """
    Simple Variational Autoencoder for compressing stock data into a latent space.
    """
    def __init__(self, input_dim=5, latent_dim=2):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2_mu = nn.Linear(16, latent_dim)
        self.fc2_logvar = nn.Linear(16, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 16)
        self.fc4 = nn.Linear(16, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

def vae_loss_function(reconstructed, original, mu, logvar):
    mse = nn.MSELoss(reduction="sum")
    recon_loss = mse(reconstructed, original)
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

def train_vae(model, data, epochs=10, batch_size=64, lr=1e-3):
    """
    Train the VAE using the provided data (numpy).
    """
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch_features = batch[0]
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch_features)
            loss = vae_loss_function(reconstructed, batch_features, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - VAE Loss: {total_loss:.2f}")


# ============ 2. (Optional) GMM for Market Regime Clustering =============== #
def train_gmm(latent_features, n_components=3):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(latent_features)
    return gmm


# ================== 3. Gym Environment (Backtesting) ======================== #
class StockTradingEnv(gym.Env):
    """
    A simplified stock trading environment where the observation is:
      - The latent representation from the VAE (and optionally GMM cluster info)
      - The current (or recent) price

    The agent takes one of three actions:
      0 = HOLD, 1 = BUY (go long), 2 = SELL (liquidate or short)

    Reward is based on changes in portfolio value over time.
    This environment simulates a single share approach for demonstration.
    """

    def __init__(
        self, 
        price_history, 
        latent_history, 
        initial_balance=10000.0,
        max_shares=10
    ):
        super(StockTradingEnv, self).__init__()

        self.price_history = price_history  # shape [T,]
        self.latent_history = latent_history  # shape [T, latent_dim or more]
        self.n_steps = len(price_history)

        self.initial_balance = initial_balance
        self.max_shares = max_shares

        # Define action & observation space
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observations: latent dim + (current price, shares held, balance ratio)
        # - You can customize this as needed
        obs_dim = self.latent_history.shape[1] + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Internal states
        self.reset()

    def _get_observation(self):
        """Construct observation from latent vector, price, shares held, balance ratio."""
        latent = self.latent_history[self.current_step]
        current_price = self.price_history[self.current_step]
        balance_ratio = self.cash / self.initial_balance
        obs = np.concatenate([
            latent, 
            [current_price, float(self.shares_held), balance_ratio]
        ])
        return obs

    def step(self, action):
        """
        Take an action (buy, sell, hold) and compute reward.
        """
        done = False
        current_price = self.price_history[self.current_step]

        # Execute action
        if action == 1:  # BUY 1 share (if possible)
            if self.cash >= current_price and self.shares_held < self.max_shares:
                self.shares_held += 1
                self.cash -= current_price
        elif action == 2:  # SELL all shares (for simplicity)
            if self.shares_held > 0:
                self.cash += self.shares_held * current_price
                self.shares_held = 0
        # If action == 0 (HOLD), do nothing

        # Move to next step
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True

        # Calculate reward as the net worth change from previous step
        net_worth = self.cash + self.shares_held * current_price
        reward = net_worth - self.prev_net_worth
        self.prev_net_worth = net_worth

        obs = self._get_observation()
        info = {"net_worth": net_worth}
        return obs, reward, done, info

    def reset(self):
        """Resets the environment to the initial state."""
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares_held = 0
        self.prev_net_worth = self.initial_balance
        return self._get_observation()


# ============ 4. Simple DQN-like Reinforcement Learning Agent =============== #
class SimpleQNetwork(nn.Module):
    """
    A minimal DQN-like network that outputs Q-values for each action.
    """
    def __init__(self, input_dim, hidden_dim=64, num_actions=3):
        super(SimpleQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x):
        return self.net(x)


def train_rl_agent(env, q_network, episodes=10, gamma=0.95, lr=1e-3, epsilon=1.0, epsilon_decay=0.95):
    """
    A toy training loop for a DQN-style agent. This uses a very simple
    on-policy approach (no replay buffer, no target network, etc.).
    In a real system, use a well-tested RL library like Stable-Baselines3.
    """
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(episodes):
        state = env.reset()
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        episode_reward = 0.0

        while not done:
            # Epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_network(state_t)
                action = q_values.argmax().item()

            next_state, reward, done, info = env.step(action)
            next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Compute target
            with torch.no_grad():
                next_q_values = q_network(next_state_t)
            max_next_q = next_q_values.max(1)[0].item()
            target_q = reward + (gamma * max_next_q if not done else 0.0)

            # Current Q
            current_q = q_network(state_t)[0, action]

            # Update Q-network
            loss = loss_fn(current_q, torch.tensor(target_q, dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Move to next step
            state_t = next_state_t
            episode_reward += reward

        # Decay epsilon
        epsilon = max(0.01, epsilon * epsilon_decay)
        print(f"Episode {ep+1}/{episodes} - Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}")


# =========================== 5. Putting It All Together ====================== #
def main():
    # ---------------- Step A: Download Data (e.g. AAPL) ----------------- #
    df = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
    df.dropna(inplace=True)
    df = df.sort_index()  # ensure chronological order

    # For simplicity, we'll just use [Open, High, Low, Close, Volume]
    features = df[["Open", "High", "Low", "Close", "Volume"]].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # --------------- Step B: Train VAE on Historical Data ----------------#
    vae = VAE(input_dim=5, latent_dim=2)
    train_vae(vae, scaled_features, epochs=5, batch_size=32, lr=1e-3)

    # Generate latent vectors for each time step
    vae.eval()
    latent_vectors = []
    with torch.no_grad():
        for row in scaled_features:
            x = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
            mu, logvar = vae.encode(x)
            z = vae.reparameterize(mu, logvar)
            latent_vectors.append(z.numpy().flatten())
    latent_vectors = np.array(latent_vectors)

    # ----- (Optional) Train GMM if you want cluster-based features ------- #
    # Here we skip it or do it quickly:
    # gmm = train_gmm(latent_vectors, n_components=3)
    # cluster_probs = gmm.predict_proba(latent_vectors)
    # or cluster_labels = gmm.predict(latent_vectors)

    # -------------- Step C: Prepare Gym Environment for RL ---------------#
    close_prices = df["Close"].values
    # For simplicity, let's directly use 'latent_vectors' as our RL observation input.
    # If you want to incorporate GMM, you'd merge cluster_probs into the observation.

    env = StockTradingEnv(price_history=close_prices, latent_history=latent_vectors)

    # -------------- Step D: Train a Simple DQN-like Agent ----------------#
    observation_dim = env.observation_space.shape[0]
    q_net = SimpleQNetwork(input_dim=observation_dim, hidden_dim=64, num_actions=3)
    train_rl_agent(env, q_net, episodes=10, gamma=0.95, lr=1e-3)

    # -------------- Step E: Evaluate the Trained Agent (Backtesting) -----#
    # We'll do a single run through the environment to see final performance
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(state_t)
        action = q_values.argmax().item()
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward

    print(f"Final net worth: {info['net_worth']:.2f}")
    print(f"Total reward earned in final run: {total_reward:.2f}")
    print("Done!")

if __name__ == "__main__":
    main()
