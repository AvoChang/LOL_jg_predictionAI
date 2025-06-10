import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler # [수정] StandardScaler 임포트

# ─── 1) Dataset 클래스 정의 (Scaler 적용 기능 추가) ────────────────────────
class LOLDataset(Dataset):
    def __init__(self, folder_path, label_max=15000.0, scaler=None): # [수정] scaler 인자 추가
        super().__init__()
        self.folder_path = folder_path
        self.label_max = label_max
        self.scaler = scaler # [수정] scaler 멤버 변수 추가

        self.file_list = [
            fname for fname in os.listdir(folder_path)
            if fname.startswith('processed_data_') and fname.endswith('.npy')
        ]

        self._data = []
        center = np.array([15000.0, 15000.0], dtype=np.float32)
        radius = 5000.0

        print("데이터 로딩 중...")
        for fname in self.file_list:
            fpath = os.path.join(folder_path, fname)
            try:
                arr = np.load(fpath, allow_pickle=True)
                for record in arr:
                    inp = record['input_data']  # (11, 110)
                    lbl = record['label']      # 원본 좌표: shape (2,)

                    if np.linalg.norm(lbl - center) <= radius:
                        continue
                    if (lbl[0] in [0, 15000]) or np.isnan(lbl).any(): continue
                    if inp.shape != (11, 110):
                        print(f"SKIP (input shape != (11,110)): {fname} -> {inp.shape}")
                        continue
                    if lbl.shape != (2,):
                        print(f"SKIP (label shape != (2,)): {fname} -> {lbl.shape}")
                        continue
                    
                    # 레이블만 정규화 (입력 데이터는 getitem에서 처리)
                    lbl_norm = lbl.astype(np.float32) / self.label_max

                    self._data.append((
                        inp.astype(np.float32),
                        lbl_norm
                    ))
            except Exception as e:
                print(f"SKIP (load error): {fname} -> {e}")
        print(f"총 {len(self._data)}개의 데이터 로드 완료.")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        inp_np, lbl_np = self._data[idx]

        # [핵심 수정] Scaler가 제공된 경우, 입력 데이터를 정규화합니다.
        if self.scaler:
            # LSTM 입력 형태 (sequence, features)에 맞게 2D로 변환하여 스케일링
            inp_np_scaled = self.scaler.transform(inp_np)
            x = torch.from_numpy(inp_np_scaled.astype(np.float32))
        else:
            x = torch.from_numpy(inp_np) # Scaler가 없으면 원본 데이터 사용

        y = torch.from_numpy(lbl_np)
        return x, y

# ─── 2) Model 정의  ───────────────────────────────────────────
class PositionPredictor(nn.Module):
    def __init__(self, input_dim=110, hidden_dim=64, dense_dim=32, output_dim=2, num_layers=1, dropout_p=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_layers,
                              batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(dense_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]
        x = self.layer_norm(last_hidden)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# ─── 3) 학습/평가 함수  ─────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip_norm=1.0):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(dataloader.dataset.indices) # Subset을 사용하므로 dataloader.dataset.indices

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
    return total_loss / len(dataloader.dataset.indices) # Subset을 사용하므로 dataloader.dataset.indices

# ─── 4) 메인 ───────────────────────────────────────────────────
if __name__ == "__main__":
    # 하이퍼파라미터
    FOLDER = "processed_data"
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    LR = 1e-4  # [수정 권장] 학습률을 약간 낮추어 안정적인 학습을 유도
    NUM_EPOCHS = 50
    GRAD_CLIP_NORM = 1.0
    DROPOUT_PROB = 0.3
    WEIGHT_DECAY = 1e-4 # 가중치 감쇠 값 증가 (e.g., 0 -> 1e-4)

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4.1) 데이터셋 로드 (정규화 없이 원본 데이터만 로드)
    full_dataset_raw = LOLDataset(FOLDER)
    
    # 데이터셋 인덱스 분할
    N = len(full_dataset_raw)
    indices = list(range(N))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # --- [핵심 수정] 입력 데이터 정규화를 위한 Scaler 정의 및 학습 ---
    print("학습 데이터 기준으로 Scaler를 학습합니다...")
    # 1. 학습 데이터의 입력(inp_np)만 모아서 2D 배열로 만듭니다.
    #    (데이터의 각 시점(row)을 독립적인 샘플로 보고 정규화)
    train_inputs_list = [full_dataset_raw._data[i][0] for i in train_idx]
    train_inputs_flat = np.vstack(train_inputs_list) # (num_samples * 11, 110) 형태

    # 2. StandardScaler를 생성하고 학습 데이터에만 fit 합니다.
    scaler = StandardScaler()
    scaler.fit(train_inputs_flat)
    print("Scaler 학습 완료.")
    # ---------------------------------------------------------------

    # 4.2) [수정] 학습/검증 데이터셋에 Scaler를 적용하여 최종 데이터셋 생성
    # 이제 LOLDataset에 scaler를 전달하여 __getitem__에서 정규화가 이뤄지도록 합니다.
    # Subset을 사용하여 동일한 원본 데이터에서 인덱스만으로 train/val을 구분합니다.
    dataset_with_scaler = LOLDataset(FOLDER, scaler=scaler)
    train_dataset = Subset(dataset_with_scaler, train_idx)
    val_dataset = Subset(dataset_with_scaler, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    # 4.3) 모델, 손실 함수, 옵티마이저
    model = PositionPredictor(input_dim=110, hidden_dim=64, dense_dim=32, output_dim=2, dropout_p=DROPOUT_PROB).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 4.4) 학습 루프
    print("\n--- 학습 시작 ---")
    best_val_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip_norm=GRAD_CLIP_NORM)
        val_loss   = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "position_predictor_best_model_final.pt")
            print(f"  -> 모델 저장 완료 (Val Loss: {val_loss:.6f})")

    print("--- 학습 완료 ---")