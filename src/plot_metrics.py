import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load CSV
df = pd.read_csv("results/training_log.csv")

# Create plots folder
Path("results/plots").mkdir(parents=True, exist_ok=True)

# ---- LOSS ----
plt.figure()
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig("results/plots/loss_curve.png")
plt.close()

# ---- DICE ------
plt.figure()
plt.plot(df["epoch"], df["train_dice"], label="Train Dice")
plt.plot(df["epoch"], df["val_dice"], label="Validation Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Training vs Validation Dice")
plt.legend()
plt.grid()
plt.savefig("results/plots/dice_curve.png")
plt.close()

print("Plots saved to results/plots/")