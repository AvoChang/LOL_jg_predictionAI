import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ─── 1) Dataset 클래스 정의 ───────────────────────────────────
class LOLDataset(Dataset):
    def __init__(self, folder_path, label_max=1500.0):
        super().__init__()
        self.folder_path = folder_path
        self.label_max = label_max

        # "processed_data_*.npy" 파일만 골라서
        self.file_list = [
            fname for fname in os.listdir(folder_path)
            if fname.startswith('processed_data_') and fname.endswith('.npy')
        ]

        self._data = []
        # 원 좌표 중심 및 반지름 설정
        center = np.array([15000.0, 15000.0], dtype=np.float32)
        radius = 5000.0

        for fname in self.file_list:
            fpath = os.path.join(folder_path, fname)
            try:
                arr = np.load(fpath, allow_pickle=True)  # 여러 레코드를 포함할 수 있음
                for record in arr:
                    inp = record['input_data']  # 기대: shape (11, 110)
                    lbl = record['label']       # 원본 좌표: shape (2,)

                    # 레이블이 지정 원 내부에 있으면 건너뛰기
                    if np.linalg.norm(lbl - center) <= radius:
                        continue

                    # 레이블이 0, 15000, np.nan 인 경우 건너뛰기 (정글러가 죽어있는 경우)
                    if (lbl[0] in [0, 15000]) or np.isnan(lbl).any(): continue

                    # 입력 데이터 형식 확인
                    if inp.shape != (11, 110):
                        print(f"SKIP (input shape != (11,110)): {fname} -> {inp.shape}")
                        continue

                    # 레이블 형식 확인
                    if lbl.shape != (2,):
                        print(f"SKIP (label shape != (2,)): {fname} -> {lbl.shape}")
                        continue

                    # 레이블 정규화: [0, label_max] -> [0,1]
                    lbl_norm = lbl.astype(np.float32) / self.label_max

                    # 데이터 저장
                    self._data.append((
                        inp.astype(np.float32),  # (11,110)
                        lbl_norm                 # (2,)
                    ))
            except Exception as e:
                print(f"SKIP (load error): {fname} -> {e}")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        inp_np, lbl_np = self._data[idx]
        x = torch.from_numpy(inp_np)  # shape: (11, 110)
        y = torch.from_numpy(lbl_np)  # shape: (2,)
        return x, y

    
# ─── 2) Model 정의 ─────────────────────────────────────────────
class PositionPredictor(nn.Module):
    def __init__(self, input_dim=110, hidden_dim=64, dense_dim=32, output_dim=2, num_layers=1, dropout_p=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        # Layer Normalization 추가: LSTM 출력에 적용하여 기울기 흐름 개선
        self.layer_norm = nn.LayerNorm(hidden_dim) # hn[-1]의 차원에 맞춰서
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p) # Dropout layer 적용
        self.fc2 = nn.Linear(dense_dim, output_dim)
        self.sigmoid = nn.Sigmoid() # Sigmoid 활성화 함수 추가

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]
        
        # Layer Normalization 적용
        x = self.layer_norm(last_hidden)
        
        x = self.fc1(x) # LayerNorm 적용된 값을 사용
        x = self.relu(x)
        x = self.dropout(x) # Dropout 적용
        x = self.fc2(x)
        x = self.sigmoid(x) # Sigmoid 적용
        return x

# ─── 3) 학습/평가 함수 ─────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip_norm=1.0): # grad_clip_norm 인자 추가
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        # Gradient Clipping 적용
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
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
    GRAD_CLIP_NORM = 1.0 # Gradient Clipping 값 설정
    DROPOUT_PROB = 0.3   # Dropout 확률 설정

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 4.1) 데이터셋 & 분할
    full_dataset = LOLDataset(FOLDER) # label_max는 기본값 15000.0 사용
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
    model = PositionPredictor(input_dim=110, hidden_dim=64, dense_dim=32, output_dim=2, dropout_p=DROPOUT_PROB).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 학습률 스케줄러 추가: 검증 손실이 개선되지 않을 때 학습률을 줄임
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 데이터 정규화 여부 확인 (이전과 동일)
    dataset_for_check = LOLDataset("processed_data", label_max=15000.0)
    all_labels = np.stack([lbl.numpy() for _, lbl in dataset_for_check], axis=0)

    print("▶ 레이블 개수:", all_labels.shape[0])
    print("▶ 정규화된 레이블 최소값 (x, y):", all_labels.min(axis=0))
    print("▶ 정규화된 레이블 최대값 (x, y):", all_labels.max(axis=0))
    print("▶ 정규화된 레이블 샘플 5개:\n", all_labels[:5])

    # 4.3) 학습 루프
    print("\n--- 학습 시작 ---")
    best_val_loss = float('inf') # 가장 좋은 검증 손실 추적
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip_norm=GRAD_CLIP_NORM)
        val_loss   = evaluate(model, val_loader, criterion, device)
        
        # 학습률 스케줄러 스텝
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # 모델 저장 로직 (선택 사항: 가장 좋은 모델만 저장)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "position_predictor_best_model_with_gvm_fixes.pt")
            print(f"모델 저장: Val Loss {val_loss:.6f} (이전 {best_val_loss:.6f} 보다 좋음)")

    print("--- 학습 완료 ---")