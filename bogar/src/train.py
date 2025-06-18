import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.preprocessing import mol_to_features, AlzheimerDataset
from src.model import HybridQuantumModel
import os

from rdkit import Chem

def train_and_save_model(csv_file_path, model_save_path):
    """
    Loads data, trains the HybridQuantumModel, and saves its state_dict.
    """
    print(f"Loading data from {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {csv_file_path}")
        return

    # Drop invalid SMILES (RDKit's MolFromSmiles returns None for invalid ones)
    df_cleaned = df[df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)].copy()
    
    print("Extracting features from SMILES...")
    df_cleaned["features"] = df_cleaned["smiles"].apply(mol_to_features)
    df_cleaned = df_cleaned[df_cleaned["features"].notnull()]

    if df_cleaned.empty:
        print("No valid SMILES entries found after cleaning and feature extraction. Cannot train model.")
        return

    X = np.array(df_cleaned["features"].tolist())
    # Ensure 'activity' column exists and is numeric
    if 'activity' not in df_cleaned.columns:
        print("Error: 'activity' column not found in the dataset. Please ensure your CSV has an 'activity' column.")
        return
    y = df_cleaned["activity"].astype(int).values

    print(f"Data shape: X={X.shape}, y={y.shape}")

    if len(np.unique(y)) < 2:
        print("Warning: Only one class found in 'activity'. Binary classification is not possible. Ensure your dataset has at least two unique activity values.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_ds = AlzheimerDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
    test_ds = AlzheimerDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False)

    model = HybridQuantumModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification

    print("Starting model training...")
    num_epochs = 100 # You can adjust this
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {total_loss:.6f}")

    print("Training complete. Evaluating on test set...")
    model.eval()
    y_preds = []
    y_true_eval = []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).squeeze()
            y_preds.extend((preds >= 0.5).cpu().numpy())
            y_true_eval.extend(yb.cpu().numpy())
    
    print("\nClassification Report on Test Set:")
    print(classification_report(y_true_eval, y_preds, zero_division=0))

    # Save the trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

if __name__ == "__main__":
    csv_path = 'data/alzheimer_drug_smiles.csv' # Adjust if your CSV is elsewhere
    model_path = 'models/hybrid_quantum_model.pth'
    train_and_save_model(csv_path, model_path)