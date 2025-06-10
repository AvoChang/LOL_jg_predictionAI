import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, dense_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(dense_dim, output_dim)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]
        last_hidden = self.layer_norm(last_hidden)
        x = self.fc1(last_hidden)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 데이터셋 로딩 코드 (LOLDataset)
class LOLDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, label_max=15000.0):
        super().__init__()
        self.folder_path = folder_path
        self.label_max = label_max
        self.file_list = [
            fname for fname in os.listdir(folder_path)
            if fname.startswith('processed_data_') and fname.endswith('.npy')
        ]
        self._data = []
        for fname in self.file_list:
            arr = np.load(os.path.join(folder_path, fname), allow_pickle=True)
            for record in arr:
                inp = record['input_data']  # (11,110)
                lbl = record['label']       # normalized (2,)
                if inp.shape == (11,110) and lbl.shape == (2,):
                    self._data.append((inp.astype(np.float32), lbl.astype(np.float32)))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        inp, lbl = self._data[idx]
        return torch.from_numpy(inp), torch.from_numpy(lbl)

# 사용자 설정
DATA_FOLDER = 'processed_data'
MODEL_PATH = 'position_predictor_best_model_final.pt'
LABEL_MAX = 15000.0
SAMPLE_INDEX = randrange(1,4998)  # 시각화할 샘플의 인덱스를 지정하세요

# 1) 데이터 로드
dataset = LOLDataset(DATA_FOLDER, label_max=LABEL_MAX)
x, y_norm = dataset[SAMPLE_INDEX]

# 2) 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PositionPredictor().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 3) 예측 수행
with torch.no_grad():
    x_input = x.unsqueeze(0).to(device)  # (1, 11, 110)
    pred_norm = model(x_input).cpu().squeeze(0).numpy()  # (2,)

# 4) denormalize
pred = pred_norm * 15000
label = y_norm.numpy()# * LABEL_MAX

# 5) 시각화
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(pred[0], pred[1], c='blue', s=50, label='Prediction')
ax.scatter(label[0], label[1], c='black', s=50, label='Label')# 예측 좌표값 텍스트로 표시
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
plt.show()
