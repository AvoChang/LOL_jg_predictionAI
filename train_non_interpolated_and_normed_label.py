import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# ─── 1) Dataset 클래스 정의 ───────────────────────────────────
class LOLDataset(Dataset):
    def __init__(self, folder_path, label_max=15000.0):
        super().__init__()
        self.folder_path = folder_path
        self.label_max = label_max

        # "processed_data_*.npy" 파일만 골라서
        self.file_list = [
            fname for fname in os.listdir(folder_path)
            if fname.startswith('processed_data_') and fname.endswith('.npy')
        ]

        self._data = []
        for fname in self.file_list:
            fpath = os.path.join(folder_path, fname)
            try:
                arr = np.load(fpath, allow_pickle=True)  # shape == (N,)
                for record in arr:
                    inp = record['input_data']  # 기대: shape (11, 110), dtype float32
                    lbl = record['label']       # 기대: shape (2,), dtype float32

                    # input_data가 (11,110)이 아니면 건너뛰기
                    if inp.shape != (11, 110):
                        print(f"SKIP (input shape != (11,110)): {fname} → {inp.shape}")
                        continue

                    # 레이블 정규화: [0,15000] → [0,1]
                    if lbl.shape != (2,):
                        print(f"SKIP (label shape != (2,)): {fname} → {lbl.shape}")
                        continue
                    lbl_norm = lbl.astype(np.float32) / self.label_max

                    self._data.append((
                        inp.astype(np.float32),   # (11, 110)
                        lbl_norm                  # (2,)
                    ))
                # end for record
            except Exception as e:
                print(f"SKIP (load error): {fname} → {e}")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        inp_np, lbl_np = self._data[idx]
        x = torch.from_numpy(inp_np)  # shape: (11, 110)
        y = torch.from_numpy(lbl_np)  # shape: (2,)
        return x, y
    
# ─── 2) Model 정의 ─────────────────────────────────────────────
class PositionPredictor(nn.Module):
    def __init__(self, input_dim=110, hidden_dim=64, dense_dim=32, output_dim=2, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_dim, output_dim)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)          
        last_hidden = hn[-1]                  
        x = self.fc1(last_hidden)             
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ─── 3) 학습/평가 함수 ─────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(dataloader.dataset)

# ─── 4) 메인 ───────────────────────────────────────────────────
if __name__ == "__main__":
    # 하이퍼파라미터
    FOLDER = "processed_data"
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    LR = 1e-3
    NUM_EPOCHS = 50

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4.1) 데이터셋 & 분할
    full_dataset = LOLDataset(FOLDER)
    N = len(full_dataset)
    indices = list(range(N))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset   = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    # 4.2) 모델, 손실 함수, 옵티마이저
    model = PositionPredictor(input_dim=110, hidden_dim=64, dense_dim=32, output_dim=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 데이터 정규화 여부 확인
    import numpy as np

    # Dataset을 이미 정의하셨으니, 그대로 로드합니다
    dataset = LOLDataset("processed_data", label_max=15000.0)

    # 모든 레이블을 모아서 NumPy 배열로 만듭니다
    all_labels = np.stack([lbl.numpy() for _, lbl in dataset], axis=0)  # shape (N, 2)

    print("▶ 레이블 개수:", all_labels.shape[0])
    print("▶ 정규화된 레이블 최소값 (x, y):", all_labels.min(axis=0))
    print("▶ 정규화된 레이블 최대값 (x, y):", all_labels.max(axis=0))
    print("▶ 정규화된 레이블 샘플 5개:\n", all_labels[:5])

    # 4.3) 학습 루프
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # 필요하다면 마지막에 모델 저장
    torch.save(model.state_dict(), "position_predictor_.pt")
