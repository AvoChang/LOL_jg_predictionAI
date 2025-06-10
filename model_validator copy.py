import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset # Dataset 임포트 추가
import os
import numpy as np
from random import randrange

# 모델 정의
class PositionPredictor(torch.nn.Module):
    def __init__(self, input_dim=110, hidden_dim=64, dense_dim=32, output_dim=2, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=num_layers,
                                  batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_dim, dense_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(dense_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid() # <--- Sigmoid 활성화 함수 추가

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1] # batch_first=True일 때 hn의 shape는 (num_layers, batch, hidden_size)
        x = self.fc1(last_hidden)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) # <--- Sigmoid 적용
        return x

# 데이터셋 로딩 코드 (LOLDataset)
class LOLDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, label_max=15000.0): # <--- 기본값 15000.0으로 변경
        super().__init__()
        self.folder_path = folder_path
        self.label_max = label_max
        self.file_list = [
            fname for fname in os.listdir(folder_path)
            if fname.startswith('processed_data_') and fname.endswith('.npy')
        ]
        self._data = []
        # 원 좌표 중심 및 반지름 설정 (이전 코드에서 누락되어 다시 추가)
        center = np.array([15000.0, 15000.0], dtype=np.float32)
        radius = 5000.0

        for fname in self.file_list:
            fpath = os.path.join(folder_path, fname)
            try:
                arr = np.load(fpath, allow_pickle=True)
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

                    # 레이블 정규화: [0, label_max] -> [0,1] <--- 정규화 로직 다시 추가
                    lbl_norm = lbl.astype(np.float32) / self.label_max

                    self._data.append((inp.astype(np.float32), lbl_norm))
            except Exception as e:
                print(f"SKIP (load error): {fname} -> {e}")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        inp, lbl = self._data[idx]
        return torch.from_numpy(inp), torch.from_numpy(lbl)

# 사용자 설정
DATA_FOLDER = 'processed_data'
MODEL_PATH = 'position_predictor_.pt' # 이 모델 파일은 Sigmoid가 적용된 모델이어야 함
LABEL_MAX = 15000.0
SAMPLE_INDEX = randrange(1, 4998) # 시각화할 샘플의 인덱스를 지정하세요

# 1) 데이터 로드
dataset = LOLDataset(DATA_FOLDER, label_max=LABEL_MAX)
x, y_norm = dataset[SAMPLE_INDEX]

# 2) 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PositionPredictor().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"모델을 성공적으로 로드했습니다: {MODEL_PATH}")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    print("모델 파일이 존재하지 않거나, 모델 정의와 일치하지 않을 수 있습니다.")
    exit() # 오류 발생 시 프로그램 종료

model.eval()

# 3) 예측 수행
with torch.no_grad():
    x_input = x.unsqueeze(0).to(device)  # (1, 11, 110)
    pred_norm = model(x_input).cpu().squeeze(0).numpy()  # (2,)

# 4) denormalize (역정규화)
pred = pred_norm * 15000  # <--- LABEL_MAX (15000)으로 곱셈 수정
label = y_norm.numpy() * LABEL_MAX # <--- LABEL_MAX (15000)으로 곱셈 적용 (y_norm은 0~1 범위이므로)

# 5) 시각화
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(pred[0], pred[1], c='blue', s=50, label='Prediction')
ax.scatter(label[0], label[1], c='black', s=50, label='Label')
ax.text(pred[0] + 150, pred[1] + 150, # 점 옆에 약간의 오프셋
        f'P: ({pred[0]:.0f}, {pred[1]:.0f})', # 정수로 표시 (원한다면 소수점 유지)
        color='blue', fontsize=10, ha='left', va='bottom')

# 실제 레이블 좌표값 텍스트로 표시
ax.text(label[0] + 150, label[1] - 150, # 점 옆에 약간의 오프셋
        f'L: ({label[0]:.0f}, {label[1]:.0f})', # 정수로 표시
        color='black', fontsize=10, ha='left', va='top')
ax.set_xlim(0, LABEL_MAX)
ax.set_ylim(0, LABEL_MAX)
ax.set_aspect('equal', 'box')
ax.set_title(f'Sample #{SAMPLE_INDEX} Prediction vs Label')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.legend()
plt.grid(True) # 그리드 추가
plt.show()