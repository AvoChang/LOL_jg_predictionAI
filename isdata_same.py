import os
import numpy as np

folder = "linearly_interpolated_processed_data"
file_list = [
    fname for fname in os.listdir(folder)
    if fname.startswith('processed_data_') and fname.endswith('.npy')
]

match_ids = []
labels_raw = []
labels_norm = []

for fname in file_list:
    fpath = os.path.join(folder, fname)
    try:
        arr = np.load(fpath, allow_pickle=True)
        record = arr[0]
        match_ids.append(record['match_id'])
        lbl = record['label'].astype(np.float32)        # 원본 레이블 (예: [x, y] 실좌표)
        labels_raw.append(lbl.copy())
        labels_norm.append(lbl / 15000.0)                 # 정규화된 레이블
    except Exception as e:
        print(f"Fail loading {fname} → {e}")

match_ids = np.array(match_ids)
labels_raw = np.stack(labels_raw, axis=0)       # shape (파일 개수, 2)
labels_norm = np.stack(labels_norm, axis=0)     # shape (파일 개수, 2)

# 1) 고유한 match_id 개수
unique_matches = np.unique(match_ids)
print("=== Match ID 현황 ===")
print("총 파일 개수:", len(file_list))
print("고유 match_id 개수:", len(unique_matches))
print("예시 고유 match_id (최대 5):", unique_matches[:5])

# 2) 원본 레이블 분포
print("\n=== 원본 레이블 분포 ===")
print("최소값 (x, y):", labels_raw.min(axis=0))
print("최대값 (x, y):", labels_raw.max(axis=0))
print("샘플 레이블 (원본) 5개:\n", labels_raw[:5])

# 3) 정규화된 레이블 분포
print("\n=== 정규화된 레이블 분포 ===")
print("최소값 (x, y):", labels_norm.min(axis=0))
print("최대값 (x, y):", labels_norm.max(axis=0))
print("샘플 레이블 (정규화) 5개:\n", labels_norm[:5])
