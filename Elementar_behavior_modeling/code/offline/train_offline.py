"""Offline behavior modeling training."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from behavior_networks import NN_motion_prediction, NN_release_rate_prediction

_OFFLINE_DIR = os.path.dirname(os.path.abspath(__file__))


def scale_field_inputs(X: np.ndarray) -> np.ndarray:
    """Flux, Frequency, Pitch, Direction -> divide by 20, 40, 180, 360. (N,4) or length-4 vector."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    out = X.copy()
    out[:, 0] = X[:, 0] / 20.0
    out[:, 1] = X[:, 1] / 40.0
    out[:, 2] = X[:, 2] / 180.0
    out[:, 3] = X[:, 3] / 360.0
    return out

# --- Config: keep EXPERIMENT_TYPE and CONFIGS consistent; weights go under model_name. ---

# One of: 'milli' | 'reconfiguration' | 'nano'
EXPERIMENT_TYPE = 'reconfiguration'
CONFIGS = {
    'milli': {
        'excel_path': os.path.join(_OFFLINE_DIR, '../../data/milli_velocity/milli_velocity_data.xlsx'),
        'sheet_name': ['train'],
        'target_col': 'Velocity',
        'network_class': NN_motion_prediction,
        'model_name': 'behavior_model_milli',
    },
    'reconfiguration': {
        'excel_path': os.path.join(_OFFLINE_DIR, '../../data/reconfiguration/release_data.xlsx'),
        'sheet_name': ['train'],
        'target_col': 'Final_Gray',
        'network_class': NN_release_rate_prediction,
        'model_name': 'behavior_model_release',
    },
    'nano': {
        'excel_path': os.path.join(_OFFLINE_DIR, '../../data/nano_velocity/nano_velocity_data.xlsx'),
        'sheet_name': ['train_AL_1050'],
        'target_col': 'Velocity',
        'network_class': NN_motion_prediction,
        'model_name': 'behavior_model_nano_AL_1050',
    },
}

config = CONFIGS[EXPERIMENT_TYPE]
excel_path = config['excel_path']
sheet_name = config['sheet_name']
target_col = config['target_col']
NetworkClass = config['network_class']
model_name = config['model_name']

print(f"\n{'='*80}")
print(f"Experiment: {EXPERIMENT_TYPE}")
print(f"Network: {NetworkClass.__name__}")
print(f"Target column: {target_col}")
print(f"{'='*80}")
print(f"\nReading data from: {excel_path}")

if isinstance(sheet_name, str):
    sheet_names = [sheet_name]
elif isinstance(sheet_name, list):
    sheet_names = sheet_name
elif not sheet_name:
    xls = pd.ExcelFile(excel_path)
    sheet_names = [xls.sheet_names[0]]
else:
    raise ValueError("sheet_name must be a string, a list of strings, or None/False")

if len(sheet_names) == 1:
    print(f"Sheet: {sheet_names[0]}")
else:
    print(f"Sheets: {sheet_names} (will be combined)")

all_dfs = []
for sn in sheet_names:
    df_sheet = pd.read_excel(excel_path, sheet_name=sn)
    df_sheet['Source_Sheet'] = sn
    print(f"  Sheet '{sn}': {len(df_sheet)} samples")
    all_dfs.append(df_sheet)

df = pd.concat(all_dfs, ignore_index=True)
print(f"\nCombined data: {len(df)} samples (from {len(sheet_names)} sheet(s))")
print(f"Columns: {df.columns.tolist()}")

def _safe_sheet_name(s: str) -> str:
    """Sanitize sheet name for filenames / history fields."""
    return str(s).strip().replace(' ', '_').replace('/', '_').replace('\\', '_')

if len(sheet_names) == 1:
    sheet_name_str = _safe_sheet_name(sheet_names[0])
else:
    sheet_name_str = '_'.join([_safe_sheet_name(sn) for sn in sheet_names])

input_cols = ['Flux', 'Frequency', 'Pitch', 'Direction']
required_cols = input_cols + [target_col]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Excel file missing required columns: {missing_cols}")

df = df.dropna(subset=required_cols)
print(f"After removing NaN values: {len(df)} samples")

if len(df) < 20:
    raise ValueError(f"Not enough data samples ({len(df)}); at least 20 valid rows required (legacy threshold).")

# Same scaling as predict_offline.
X = df[input_cols].values
X_scaled = scale_field_inputs(X)

print('\nX_scaled (first 5 rows):')
print(X_scaled[:5])
print(f'X_scaled dtype: {X_scaled.dtype}')
print(f'X_scaled min: {X_scaled.min(axis=0)}')
print(f'X_scaled max: {X_scaled.max(axis=0)}')

y = df[target_col].values.reshape(-1, 1)

print(f"\nInput shape: {X_scaled.shape}")
print(f"Output shape: {y.shape}")

X_train = X_scaled
y_train = y

print(f"\nDataset:")
print(f"  Train: {len(X_train)} samples (100.0%)")

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)

min_training_samples = 8
if len(X_train) < min_training_samples:
    raise ValueError(f"Not enough training samples ({len(X_train)}). Need at least {min_training_samples} samples.")

batch_size = 128  # Same default as train_from_excel_v2
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

model = NetworkClass(input_dim=4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

epochs = 2000  # Matches train_from_excel_v2 MAX_EPISODES; reduce manually for shorter runs
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

train_losses = []

print("\nStarting training...")
print("=" * 60)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    batch_count = 0

    for batch_X, batch_y in train_loader:
        # Skip tiny batches for stable BatchNorm (same idea as train_from_excel_v2)
        if batch_X.shape[0] < 8:
            continue
        
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        batch_count += 1
    
    if batch_count > 0:
        avg_train_loss = train_loss / batch_count
        train_losses.append(avg_train_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}")

print("=" * 60)
print("Training completed!")

# Save next to the data Excel under train_output/ for one dataset per folder.
excel_dir = os.path.dirname(os.path.abspath(excel_path))
train_output_dir = os.path.join(excel_dir, 'train_output')
os.makedirs(train_output_dir, exist_ok=True)

model_dir = os.path.join(train_output_dir, model_name)
os.makedirs(model_dir, exist_ok=True)
print(f"\nModel folder: {model_dir}")

model_filename = f'{model_name}.pth'
model_save_path = os.path.join(model_dir, model_filename)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to: {model_save_path}")

train_data_filename = f'train_data_{model_name}.xlsx'
train_data_path = os.path.join(model_dir, train_data_filename)
train_data_cols = required_cols + (['Source_Sheet'] if 'Source_Sheet' in df.columns else [])
train_data_df = df[train_data_cols].copy()
train_data_df.to_excel(train_data_path, index=False)
print(f"Training data saved to: {train_data_path}")
print(f"  - Total samples: {len(train_data_df)}")
if len(sheet_names) > 1 and 'Source_Sheet' in train_data_df.columns:
    print(f"  - Source sheets: {sheet_names}")
    for sn in sheet_names:
        count = int((train_data_df['Source_Sheet'] == sn).sum())
        print(f"    - '{sn}': {count} samples")

history = {
    'train_losses': train_losses,
    'model_name': model_name,
    'experiment_type': EXPERIMENT_TYPE,
    'sheet_names': sheet_names,
    'sheet_name_str': sheet_name_str,
    'excel_path': excel_path,
    'target_col': target_col,
    'num_samples': len(df),
    'epochs': epochs,
    'batch_size': batch_size,
    'final_train_loss': train_losses[-1] if train_losses else None
}
history_filename = f'training_history_{model_name}.npy'
history_path = os.path.join(model_dir, history_filename)
np.save(history_path, history)
print(f"Training history saved to: {history_path}")

print("\nTraining summary:")
print(f"  Experiment type: {EXPERIMENT_TYPE}")
print(f"  Network: {NetworkClass.__name__}")
print(f"  Target column: {target_col}")
print(f"  Final Train Loss: {train_losses[-1]:.6f}")
print(f"  All files saved to folder: {model_dir}")
