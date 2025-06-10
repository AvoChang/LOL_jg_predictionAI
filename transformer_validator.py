import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from random import randrange
import pickle

# ─── 1) 학습에 사용했던 모델과 데이터셋 클래스를 그대로 정의 ───────────────────

# 1.1) LOLDataset 클래스 (Scaler를 사용하는 버전)
class LOLDataset(Dataset):
    def __init__(self, folder_path, scaler=None, label_max=15000.0):
        super().__init__()
        self.scaler = scaler
        self.label_max = label_max
        
        # 데이터 캐싱으로 반복 로딩 방지
        if not hasattr(LOLDataset, '_global_data_cache'):
            print("데이터 로딩 중...")
            _global_data_cache = []
            file_list = [f for f in os.listdir(folder_path) if f.startswith('processed_data_') and f.endswith('.npy')]
            center = np.array([15000.0, 15000.0], dtype=np.float32)
            radius = 5000.0
            for fname in file_list:
                fpath = os.path.join(folder_path, fname)
                try:
                    arr = np.load(fpath, allow_pickle=True)
                    for record in arr:
                        inp, lbl = record['input_data'], record['label']
                        if np.linalg.norm(lbl - center) <= radius or (lbl[0] in [0, 15000]) or np.isnan(lbl).any(): continue
                        if inp.shape != (11, 110) or lbl.shape != (2,): continue
                        
                        _global_data_cache.append((inp.astype(np.float32), lbl.astype(np.float32)))
                except Exception as e:
                    print(f"SKIP (load error): {fname} -> {e}")
            LOLDataset._global_data_cache = _global_data_cache
            print(f"총 {len(LOLDataset._global_data_cache)}개의 데이터 로드 완료.")
        
        self._data = LOLDataset._global_data_cache

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        inp_np, lbl_np = self._data[idx]
        
        # 입력 데이터 정규화 (Scaler 사용)
        if self.scaler:
            inp_np = self.scaler.transform(inp_np)
        
        # 레이블 데이터 정규화
        lbl_norm_np = lbl_np / self.label_max

        x = torch.from_numpy(inp_np.astype(np.float32))
        y_norm = torch.from_numpy(lbl_norm_np.astype(np.float32))
        
        # 시각화를 위해 원본 레이블도 함께 반환
        return x, y_norm, torch.from_numpy(lbl_np)

# 1.2) Transformer 모델 클래스
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=110, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=512, dropout=0.2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid()
        )
        self.d_model = d_model

    def forward(self, src):
        src = self.input_projection(src)
        cls_tokens = self.cls_token.expand(src.size(0), -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        cls_output = output[:, 0, :]
        return self.output_head(cls_output)

# ─── 2) 검증 및 시각화 실행 ───────────────────────────────────────────────

if __name__ == "__main__":
    # --- 사용자 설정 ---
    DATA_FOLDER = 'processed_data'
    MODEL_PATH = 'transformer_predictor_best_model.pt' # 트랜스포머 모델 경로
    SCALER_PATH = 'scaler.pkl' # 저장된 Scaler 경로
    LABEL_MAX = 15000.0
    SAMPLE_INDEX = randrange(0, 5000)  # 시각화할 샘플 인덱스 (0부터 시작)
    # -----------------

    # 1) Scaler 로드
    print(f"'{SCALER_PATH}'에서 Scaler 로딩...")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # 2) 데이터셋 로드 (Scaler 전달)
    dataset = LOLDataset(DATA_FOLDER, scaler=scaler, label_max=LABEL_MAX)
    
    # 인덱스 범위 확인
    if SAMPLE_INDEX >= len(dataset):
        print(f"Error: SAMPLE_INDEX({SAMPLE_INDEX})가 데이터셋 크기({len(dataset)})를 벗어납니다. 인덱스를 0과 {len(dataset)-1} 사이로 설정해주세요.")
    else:
        # 특정 인덱스의 데이터 가져오기
        x, y_norm, y_original = dataset[SAMPLE_INDEX]

        # 3) 모델 로드
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 학습 시 사용했던 하이퍼파라미터와 동일하게 모델 초기화
        model = TransformerPredictor(
            input_dim=110, d_model=128, nhead=8, num_encoder_layers=3, 
            dim_feedforward=512, dropout=0.2
        ).to(device)
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"'{MODEL_PATH}'에서 모델 로딩 완료.")

        # 4) 예측 수행
        with torch.no_grad():
            # 모델 입력에 맞게 배치 차원 추가
            x_input = x.unsqueeze(0).to(device) 
            # 모델 예측 (결과는 정규화된 상태)
            pred_norm = model(x_input).cpu().squeeze(0).numpy()

        # 5) 결과값 역정규화 (Denormalize)
        pred_denorm = pred_norm * LABEL_MAX
        label_denorm = y_original.numpy() # 데이터셋에서 바로 받은 원본 레이블 사용

        # 6) 시각화
        print(f"Sample #{SAMPLE_INDEX} 시각화...")
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 예측 위치와 실제 위치를 점으로 표시
        ax.scatter(pred_denorm[0], pred_denorm[1], c='blue', s=80, label='Prediction', alpha=0.8, marker='X')
        ax.scatter(label_denorm[0], label_denorm[1], c='red', s=80, label='Ground Truth', alpha=0.8, marker='o')

        # 예측 좌표 텍스트
        ax.text(pred_denorm[0] + 200, pred_denorm[1] + 200,
                f'Pred: ({pred_denorm[0]:.0f}, {pred_denorm[1]:.0f})',
                color='blue', fontsize=10, ha='left', va='bottom', weight='bold')

        # 실제 좌표 텍스트
        ax.text(label_denorm[0] + 200, label_denorm[1] - 200,
                f'Label: ({label_denorm[0]:.0f}, {label_denorm[1]:.0f})',
                color='red', fontsize=10, ha='left', va='top', weight='bold')
        
        # 거리 계산 및 표시
        distance = np.linalg.norm(pred_denorm - label_denorm)
        ax.plot([pred_denorm[0], label_denorm[0]], [pred_denorm[1], label_denorm[1]], 'k--', alpha=0.7)
        ax.set_title(f'Sample #{SAMPLE_INDEX} | Prediction vs. Ground Truth\nDistance: {distance:.2f}', fontsize=14)

        ax.set_xlim(0, LABEL_MAX)
        ax.set_ylim(0, LABEL_MAX)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X coordinate', fontsize=12)
        ax.set_ylabel('Y coordinate', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=12)
        
        # 소환사의 협곡 이미지 배경으로 추가 (선택 사항)
        try:
            # 아래 'map_image.png' 파일이 코드와 같은 경로에 있어야 합니다.
            # (이미지 파일은 인터넷에서 '소환사의 협곡 미니맵' 등으로 검색하여 구할 수 있습니다)
            map_img = plt.imread('map_image.png')
            ax.imshow(map_img, extent=[0, LABEL_MAX, 0, LABEL_MAX], aspect='auto', alpha=0.5)
        except FileNotFoundError:
            print("배경 이미지('map_image.png')를 찾을 수 없습니다. 좌표만 표시합니다.")

        plt.show()