"""
=============================================================================
 VAE-GMM Stock Prediction Model
-----------------------------------------------------------------------------
 This file implements an end-to-end pipeline for:
   1. Data Ingestion & Preprocessing
   2. Variational Autoencoder (VAE) Training
   3. Gaussian Mixture Model (GMM) Clustering
   4. Probability Prediction & Buy/Sell Signal Generation
   5. Backtesting
   6. (Optional) Real-time Inference Pipeline

 Reference: VAE-GMM Stock Prediction Model Specification Document
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ======================= Section 1: Data Ingestion ===========================
# Ref. Specification 2.1: "Data Ingestion: The system should collect OHLCV 
# price data, fundamentals, and macroeconomic indicators."


class StockDataset(Dataset):
    """
    A custom PyTorch dataset to handle stock time series and fundamental data.
    Reference: Specification 3.1 - Data Sources & 3.2 - Data Preprocessing
    """

    def __init__(self, tickers, start_date, end_date, transform=None):
        """
        Args:
            tickers (list): List of stock tickers to fetch.
            start_date (str): Start date for historical data (YYYY-MM-DD).
            end_date (str): End date for historical data (YYYY-MM-DD).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.transform = transform

        self.data_frames = []
        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start_date, end=self.end_date)
            df["Ticker"] = ticker
            self.data_frames.append(df)

        # Combine all data into a single DataFrame
        self.data = pd.concat(self.data_frames)
        self.data.sort_index(inplace=True)
        
        # Basic cleaning of missing values (forward fill)
        self.data.fillna(method="ffill", inplace=True)

        # For demonstration, we are only focusing on OHLCV. 
        # In a real scenario, fundamental and macro data would be merged here.

        # Convert DataFrame to a list of samples if needed
        self.samples = self.data.reset_index(drop=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Example features: [Open, High, Low, Close, Volume]
        row = self.samples.iloc[idx]
        features = np.array([row["Open"], row["High"], row["Low"],
                             row["Close"], row["Volume"]], dtype=np.float32)
        
        # (Optional) apply transformations if needed
        if self.transform:
            features = self.transform(features)

        return features


# ================== Section 2: Data Preprocessing & Utilities ===============
# Ref. Specification 3.2: "Data Preprocessing Steps: Normalize numerical data, 
# log-scale, handle missing, etc."


def preprocess_data(dataset):
    """
    Normalize and preprocess the dataset as per specification:
      - Normalize numerical data
      - Handle missing values
      - Perform optional feature engineering
    """
    scaler = StandardScaler()
    all_features = []
    
    # Convert the entire dataset to a NumPy array
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for batch in loader:
        features = batch.numpy()
        all_features.append(features)
    
    # Stack into single array
    all_features = np.vstack(all_features)
    scaled_features = scaler.fit_transform(all_features)
    
    return scaled_features, scaler


# ===================== Section 3: Variational Autoencoder ====================
# Ref. Specification 4.1: "VAE Model: Train a Variational Autoencoder (VAE)
# to extract latent features."

class VAE(nn.Module):
    """
    A simple Variational Autoencoder for stock time series as described in
    Specification 4.1.
    """

    def __init__(self, input_dim=5, latent_dim=2):
        """
        Args:
            input_dim (int): Number of input features (e.g., OHLCV).
            latent_dim (int): Dimensionality of the latent space.
        """
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
    """
    Combination of Reconstruction Loss (MSE) and KL Divergence.
    Reference: Specification 4.1.
    """
    mse = nn.MSELoss(reduction="sum")
    recon_loss = mse(reconstructed, original)

    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kld


def train_vae(model, data, epochs=10, batch_size=64, lr=1e-3):
    """
    Train the VAE using the provided dataset.
    
    Args:
        model (VAE): An instance of the VAE model.
        data (np.ndarray): Preprocessed data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
    """
    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_features in dataloader:
            batch_features = batch_features[0]
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch_features)
            loss = vae_loss_function(reconstructed, batch_features, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")


# ==================== Section 4: Gaussian Mixture Model ======================
# Ref. Specification 4.2: "GMM Model: Train a Gaussian Mixture Model (GMM) 
# to cluster the latent space into different market regimes."

def train_gmm(latent_features, n_components=3):
    """
    Train a Gaussian Mixture Model on the latent features.
    
    Args:
        latent_features (np.ndarray): Latent representations from the VAE.
        n_components (int): Number of mixture components.
        
    Returns:
        gmm (GaussianMixture): Trained GMM model.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(latent_features)
    return gmm


# ============= Section 5: Buy/Sell Decision Mechanism & Backtesting ==========
# Ref. Specification 4.3: "If Bullish Probability > 70% -> BUY, etc."

def generate_signals(probabilities, bullish_index=0, threshold=0.7):
    """
    Generate buy/sell/hold signals based on probability distribution from GMM.

    Args:
        probabilities (np.ndarray): Probabilities for each GMM component.
        bullish_index (int): Index of the GMM component considered "bullish".
        threshold (float): Probability threshold for buy/sell signals.

    Returns:
        signals (list): List of signals ("BUY", "SELL", or "HOLD").
    """
    signals = []
    for prob in probabilities:
        # Example: If cluster '0' is bullish, cluster '1' might be bearish, etc.
        bullish_prob = prob[bullish_index]  # e.g., first cluster as bullish
        if bullish_prob > threshold:
            signals.append("BUY")
        elif bullish_prob < (1 - threshold):
            signals.append("SELL")
        else:
            signals.append("HOLD")
    return signals


def backtest(signals, price_data):
    """
    A simple backtesting mechanism to calculate returns based on signals.
    Reference: Specification 5.1 & 5.2 (Sharpe, returns, etc.).
    
    Args:
        signals (list): Generated signals for each time step.
        price_data (np.ndarray): Corresponding price data (e.g., close prices).
    
    Returns:
        performance_dict (dict): A dictionary with example performance metrics.
    """
    initial_investment = 10000.0
    cash = initial_investment
    holdings = 0
    last_price = 0.0

    # For simplicity, assume one share per trade
    for i, signal in enumerate(signals):
        if signal == "BUY" and cash > price_data[i]:
            # Buy 1 share
            holdings += 1
            cash -= price_data[i]
            last_price = price_data[i]
        elif signal == "SELL" and holdings > 0:
            # Sell all shares
            cash += holdings * price_data[i]
            holdings = 0
            last_price = price_data[i]
        # HOLD does nothing
    
    # Liquidate any remaining holdings at final price
    if holdings > 0:
        cash += holdings * last_price
        holdings = 0

    final_value = cash
    return_ = (final_value - initial_investment) / initial_investment

    performance_dict = {
        "Initial Investment": initial_investment,
        "Final Value": final_value,
        "Total Return (%)": return_ * 100.0,
    }

    return performance_dict


# ==================== Section 6: Putting It All Together =====================

def main():
    """
    Main entry point for training and inference pipeline.
    Reference: Specification 6.1 & 6.2 for Deployment and Real-time integration.
    """

    # ------------------------- Step 1: Data Ingestion ------------------------
    tickers = ["AAPL"]  # Just one ticker for demo; can be extended to S&P 500
    dataset = StockDataset(
        tickers=tickers,
        start_date="2020-01-01",
        end_date="2021-01-01",
    )

    # --------------------- Step 2: Data Preprocessing -----------------------
    scaled_data, scaler = preprocess_data(dataset)

    # -------------------- Step 3: Train the VAE for Latent Features ---------
    vae = VAE(input_dim=5, latent_dim=2)
    train_vae(model=vae, data=scaled_data, epochs=5, batch_size=32, lr=1e-3)

    # Obtain latent features
    vae.eval()
    with torch.no_grad():
        latent_list = []
        for row in scaled_data:
            row_torch = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
            mu, logvar = vae.encode(row_torch)
            z = vae.reparameterize(mu, logvar)
            latent_list.append(z.numpy().squeeze())
    latent_features = np.array(latent_list)

    # --------------------- Step 4: Train the GMM on Latent ------------------
    gmm = train_gmm(latent_features, n_components=3)

    # Example assumption: cluster 0 is bullish, 1 is bearish, 2 is neutral
    probabilities = gmm.predict_proba(latent_features)
    signals = generate_signals(probabilities, bullish_index=0, threshold=0.7)

    # ------------------------ Step 5: Backtesting ---------------------------
    # For demonstration, we’ll just use the Close price from the dataset
    # Note: Ensure indices match properly with your data alignment
    price_data = dataset.data["Close"].values[:len(signals)]
    performance = backtest(signals, price_data)
    print("Backtesting Performance:")
    for k, v in performance.items():
        print(f"{k}: {v}")

    # --------------------- Step 6: Real-time / Next Steps -------------------
    # Integration with a live pipeline would require:
    #   1. Fetching new data in real-time
    #   2. Preprocessing with the same scaler
    #   3. Getting latent features from the VAE
    #   4. Predicting cluster probabilities via GMM
    #   5. Generating signals for each new data point
    print("\nReal-time integration would involve repeating these steps with latest data.")


if __name__ == "__main__":
    main()
