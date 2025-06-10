import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

# ─── 1) Dataset 클래스 정의 (변경 없음) ───────────────────────────────
class LOLDataset(Dataset):
    def __init__(self, folder_path, label_max=15000.0, scaler=None):
        super().__init__()
        self.folder_path = folder_path
        self.label_max = label_max
        self.scaler = scaler

        self.file_list = [
            fname for fname in os.listdir(folder_path)
            if fname.startswith('processed_data_') and fname.endswith('.npy')
        ]

        self._data = []
        center = np.array([15000.0, 15000.0], dtype=np.float32)
        radius = 5000.0

        # 데이터 로딩은 한 번만 수행하도록 개선
        if not hasattr(LOLDataset, '_global_data_cache'):
            print("데이터 로딩 중...")
            _global_data_cache = []
            for fname in self.file_list:
                fpath = os.path.join(folder_path, fname)
                try:
                    arr = np.load(fpath, allow_pickle=True)
                    for record in arr:
                        inp = record['input_data']
                        lbl = record['label']

                        if np.linalg.norm(lbl - center) <= radius: continue
                        if (lbl[0] in [0, 15000]) or np.isnan(lbl).any(): continue
                        if inp.shape != (11, 110): continue
                        if lbl.shape != (2,): continue
                        
                        lbl_norm = lbl.astype(np.float32) / self.label_max
                        _global_data_cache.append((inp.astype(np.float32), lbl_norm))
                except Exception as e:
                    print(f"SKIP (load error): {fname} -> {e}")
            LOLDataset._global_data_cache = _global_data_cache
            print(f"총 {len(LOLDataset._global_data_cache)}개의 데이터 로드 완료.")

        self._data = LOLDataset._global_data_cache
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        inp_np, lbl_np = self._data[idx]
        if self.scaler:
            inp_np_scaled = self.scaler.transform(inp_np)
            x = torch.from_numpy(inp_np_scaled.astype(np.float32))
        else:
            x = torch.from_numpy(inp_np)
        y = torch.from_numpy(lbl_np)
        return x, y

# ─── 2) Transformer Model 정의 ───────────────────────────────────

# 2.1) Positional Encoding 모듈
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 2.2) Transformer 기반 예측 모델
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=110, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=256, dropout=0.3):
        super().__init__()
        
        # 1. 입력 프로젝션: 입력 피처(110)를 트랜스포머의 d_model(128) 차원으로 변환
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 2. CLS 토큰: 시퀀스 전체 정보를 요약하기 위한 학습 가능한 토큰
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 3. 위치 인코딩
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 4. 트랜스포머 인코더
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 5. 최종 예측을 위한 MLP (분류 헤드)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid() # 레이블이 [0,1] 범위이므로 Sigmoid 사용
        )
        self.d_model = d_model

    def forward(self, src):
        # src: (batch_size, seq_len=11, input_dim=110)
        
        # 1. 입력 프로젝션
        src = self.input_projection(src) # (batch_size, 11, d_model)
        
        # 2. CLS 토큰 추가
        cls_tokens = self.cls_token.expand(src.size(0), -1, -1) # (batch_size, 1, d_model)
        src = torch.cat((cls_tokens, src), dim=1) # (batch_size, 12, d_model)
        
        # 3. 위치 인코딩 적용
        src = self.pos_encoder(src)
        
        # 4. 트랜스포머 인코더 통과
        output = self.transformer_encoder(src) # (batch_size, 12, d_model)
        
        # 5. CLS 토큰에 해당하는 출력만 사용
        cls_output = output[:, 0, :] # (batch_size, d_model)
        
        # 6. MLP 헤드를 통해 최종 위치 예측
        return self.output_head(cls_output)


# ─── 3) 학습/평가 함수 (변경 없음) ────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip_norm=1.0):
    model.train()
    total_loss = 0.0
    num_samples = len(dataloader.dataset)
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
    return total_loss / num_samples

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_samples = len(dataloader.dataset)
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            total_loss += loss.item() * batch_x.size(0)
    return total_loss / num_samples

# ─── 4) 메인 ───────────────────────────────────────────────────
if __name__ == "__main__":
    # 하이퍼파라미터
    FOLDER = "processed_data"
    BATCH_SIZE = 16 # Transformer는 더 많은 메모리를 사용할 수 있으므로 배치 사이즈 조절 가능
    NUM_WORKERS = 2
    LR = 1e-4
    NUM_EPOCHS = 50
    GRAD_CLIP_NORM = 1.0

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터셋 로드 및 분할
    full_dataset_raw = LOLDataset(FOLDER)
    indices = list(range(len(full_dataset_raw)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # Scaler 학습
    print("학습 데이터 기준으로 Scaler를 학습합니다...")
    train_inputs_list = [full_dataset_raw._data[i][0] for i in train_idx]
    train_inputs_flat = np.vstack(train_inputs_list)
    scaler = StandardScaler()
    scaler.fit(train_inputs_flat)
    print("Scaler 학습 완료.")

    # Scaler를 적용한 최종 데이터셋 생성
    dataset_with_scaler = LOLDataset(FOLDER, scaler=scaler)
    train_dataset = Subset(dataset_with_scaler, train_idx)
    val_dataset = Subset(dataset_with_scaler, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 4.3) [수정] 트랜스포머 모델, 손실 함수, 옵티마이저
    model = TransformerPredictor(
        input_dim=110,
        d_model=128,            # 트랜스포머의 주요 차원
        nhead=8,                # Multi-head Attention 헤드 수 (d_model의 약수여야 함)
        num_encoder_layers=3,   # 인코더 레이어 수
        dim_feedforward=512,    # 피드포워드 네트워크 차원
        dropout=0.2
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5) # 약간의 L2 정규화 추가
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 4.4) 학습 루프
    print("\n--- 트랜스포머 모델 학습 시작 ---")
    best_val_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip_norm=GRAD_CLIP_NORM)
        val_loss   = evaluate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
        
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "transformer_predictor_best_model.pt")
            print(f"  -> 모델 저장 완료 (Val Loss: {val_loss:.6f})")

    print("--- 학습 완료 ---")

    print("학습에 사용된 Scaler를 저장합니다...")
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler가 'scaler.pkl' 파일로 성공적으로 저장되었습니다.")