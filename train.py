import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import joblib
import json

# -------- CONFIG (Updated for a single folder structure) --------
WFA_BOYS_FILE = "tab_wfa_boys_p_0_5.xlsx"
WFA_GIRLS_FILE = "tab_wfa_girls_p_0_5.xlsx"
HFA_BOYS_FILE = "tab_hfa_boys_p_0_5.xlsx"
HFA_GIRLS_FILE = "tab_hfa_girls_p_0_5.xlsx"
BFA_FILE = "bmi.csv.xlsx"

MODEL_SAVE_PATH = "growth_model.pth"
SCALER_SAVE_PATH = "scaler.joblib"
PARAMS_SAVE_PATH = "best_params.json"
# -----------------------------------------------------------

# Training parameters
EPOCHS = 200
PATIENCE = 15
OPTUNA_TRIALS = 50
CLASS_LABELS = {0:"Underweight", 1:"Healthy", 2:"Overweight", 3:"Obese", 4:"Stunted", 5:"Normal Ht"}

# -------- AI MODEL DEFINITION for Optuna --------
class GrowthNet(nn.Module):
    def __init__(self, n_layers=2, n_units=64, dropout_rate=0.3):
        super().__init__()
        layers = []
        in_features = 4
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, n_units)); layers.append(nn.ReLU()); layers.append(nn.Dropout(dropout_rate))
            in_features = n_units
        layers.append(nn.Linear(in_features, len(CLASS_LABELS)))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

# -------- DATA UTILITIES --------
def load_who_reference(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    age_col = next(c for c in df.columns if re.search(r'age|day|month', str(c), re.I))
    pcols = [c for c in df.columns if re.match(r"P\d+", str(c))]
    df = df[[age_col] + pcols].copy()
    df.columns = ["age_days"] + pcols
    df["age_months"] = df["age_days"] / 30.4375
    return df

def build_dataset() -> pd.DataFrame:
    print("Building enhanced dataset with jitter...")
    wfa_boys = load_who_reference(WFA_BOYS_FILE); wfa_girls = load_who_reference(WFA_GIRLS_FILE)
    hfa_boys = load_who_reference(HFA_BOYS_FILE); hfa_girls = load_who_reference(HFA_GIRLS_FILE)
    bfa_df = pd.read_excel(BFA_FILE)
    
    dataset = []
    for sex, wfa, hfa in [("M", wfa_boys, hfa_boys), ("F", wfa_girls, hfa_girls)]:
        for i, row in wfa.iterrows():
            age = row["age_months"]
            for col in [c for c in wfa.columns if c.startswith("P")]:
                perc = float(re.findall(r"\d+", col)[0])
                wt = row[col]; wt_jitter = wt * np.random.normal(1, 0.02)
                if perc < 3: wfa_lbl = 0
                elif perc < 85: wfa_lbl = 1
                elif perc < 97: wfa_lbl = 2
                else: wfa_lbl = 3
                hfa_lbl = 4 if perc < 3 else 5
                try: ht = hfa.iloc[i][col]
                except (KeyError, IndexError): ht = 50 + 0.5 * age
                ht_jitter = ht * np.random.normal(1, 0.02)
                final_label = wfa_lbl if wfa_lbl != 1 else hfa_lbl
                dataset.append([age, ht_jitter, wt_jitter, 1 if sex == "M" else 0, final_label])

    for _, row in bfa_df.iterrows():
        bmi_class = str(row["BmiClass"]).lower()
        if "under" in bmi_class: lbl = 0
        elif "over" in bmi_class: lbl = 2
        elif "obese" in bmi_class: lbl = 3
        else: lbl = 1
        dataset.append([row["Age"], row["Height in centimeter"], row["Weight"], 1, lbl])

    print("Dataset built.")
    return pd.DataFrame(dataset, columns=["age", "height", "weight", "sex", "label"])

# -------- OPTUNA OBJECTIVE FUNCTION --------
def objective(trial, X_train, y_train, X_val, y_val, class_weights_tensor):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_units = trial.suggest_int("n_units", 32, 128, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    model = GrowthNet(n_layers=n_layers, n_units=n_units, dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    best_val_loss = float('inf'); epochs_no_improve = 0
    for epoch in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        outputs = model(X_train); loss = criterion(outputs, y_train)
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val_outputs = model(X_val); val_loss = criterion(val_outputs, y_val)
        if val_loss < best_val_loss: best_val_loss = val_loss; epochs_no_improve = 0
        else: epochs_no_improve += 1
        if epochs_no_improve == PATIENCE: break
    return best_val_loss

# -------- MAIN TRAINING SCRIPT --------
if __name__ == "__main__":
    data = build_dataset()
    X = data.drop("label", axis=1).values
    y = data["label"].values
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Data scaler fitted and saved to '{SCALER_SAVE_PATH}'.")

    present_classes = np.unique(y_train)
    weights_array = compute_class_weight('balanced', classes=present_classes, y=y_train)
    class_weights_tensor = torch.ones(len(CLASS_LABELS), dtype=torch.float32)
    for i, class_idx in enumerate(present_classes):
        class_weights_tensor[class_idx] = torch.tensor(weights_array[i], dtype=torch.float32)
    print("Calculated class weights to handle data imbalance.")

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, class_weights_tensor), n_trials=OPTUNA_TRIALS)
    print("\nOptuna study finished."); print("Best trial:", study.best_trial.params)

    best_params = study.best_trial.params
    with open(PARAMS_SAVE_PATH, 'w') as f:
        json.dump(best_params, f)
    print(f"✅ Best hyperparameters saved to '{PARAMS_SAVE_PATH}'.")

    print("\nTraining final model with best hyperparameters...")
    final_model = GrowthNet(n_layers=best_params['n_layers'], n_units=best_params['n_units'], dropout_rate=best_params['dropout_rate'])
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    X_train_val_tensor = torch.cat((X_train_tensor, X_val_tensor))
    y_train_val_tensor = torch.cat((y_train_tensor, y_val_tensor))
    for epoch in range(EPOCHS):
        final_model.train(); optimizer.zero_grad()
        outputs = final_model(X_train_val_tensor); loss = criterion(outputs, y_train_val_tensor)
        loss.backward(); optimizer.step()
        if (epoch + 1) % 20 == 0: print(f"Final training, Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Final optimized model saved to '{MODEL_SAVE_PATH}'.")

    print("\n--- Final Optimized Model Evaluation ---")
    final_model.eval()
    with torch.no_grad():
        y_pred_tensor = final_model(X_test_tensor)
        y_pred = torch.argmax(y_pred_tensor, dim=1).numpy()
    
    print("\nClassification Report:"); print(classification_report(y_test, y_pred, labels=list(CLASS_LABELS.keys()), target_names=list(CLASS_LABELS.values()), zero_division=0))
    print("Confusion Matrix:"); cm = confusion_matrix(y_test, y_pred, labels=list(CLASS_LABELS.keys()))
    plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_LABELS.values(), yticklabels=CLASS_LABELS.values())
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix'); plt.show()