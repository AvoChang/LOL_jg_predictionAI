import numpy as np

# 데이터 파일 경로
data_file = "linearly_interpolated_processed_data/processed_data_20250518_062632.npy"

# 데이터 로드
data = np.load(data_file)

# 데이터 출력
print("데이터 형태:", data.shape)
print("\n데이터 내용:")
for i, value in enumerate(data):
    print(f"인덱스 {i}: {value}") 