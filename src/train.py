import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

from data.dataset import X, y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

target_col = "Machine failure"
y_binary = y[target_col].values

cat_col = "Type"
label_enc = LabelEncoder()
type_encoded = label_enc.fit_transform(X[cat_col].values).reshape(-1, 1)

num_cols = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
scaler = StandardScaler()
num_scaled = scaler.fit_transform(X[num_cols].values)

features = np.concatenate([type_encoded, num_scaled], axis=1)
n_features = features.shape[1]

features_t = torch.tensor(features, dtype=torch.float32)
targets_t = torch.tensor(y_binary, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(features_t, targets_t)

n_total = len(dataset)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val
torch.manual_seed(42)
train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            nonlinearity="relu",
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


model = RNNClassifier(input_size=1).to(device)

pos_weight = torch.tensor([(n_train - y_binary[:n_train].sum()) / y_binary[:n_train].sum()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 50
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.unsqueeze(-1).to(device)
        batch_y = batch_y.to(device)

        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.unsqueeze(-1).to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            val_loss += loss.item() * batch_x.size(0)
            probs = torch.sigmoid(pred)
            predicted = (probs >= 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc = correct / total

    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

model.eval()
test_loss = 0.0
correct = 0
total = 0
all_preds = []
all_targets = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.unsqueeze(-1).to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        test_loss += loss.item() * batch_x.size(0)
        probs = torch.sigmoid(pred)
        predicted = (probs >= 0.5).float()
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())
test_loss /= len(test_loader.dataset)
test_acc = correct / total

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
p = precision_score(all_targets, all_preds, zero_division=0)
r = recall_score(all_targets, all_preds, zero_division=0)
f1 = f1_score(all_targets, all_preds, zero_division=0)
cm = confusion_matrix(all_targets, all_preds)

print(f"\nTest  loss={test_loss:.4f}  acc={test_acc:.4f}  prec={p:.4f}  rec={r:.4f}  f1={f1:.4f}")
print(f"Confusion matrix:\n{cm}")
