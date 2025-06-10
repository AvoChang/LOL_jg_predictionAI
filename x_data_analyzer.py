import numpy as np

def analyze_x_data(file_path):
    # 데이터 로드
    data = np.load(file_path)
    
    print("=== X 데이터 전체 내용 ===")
    print(f"데이터 형태: {data.shape}")
    
    # 모든 데이터 출력
    print("\n전체 데이터:")
    for i in range(data.shape[0]):  # 배치
        for j in range(data.shape[1]):  # 채널
            print(f"\n배치 {i}, 채널 {j}:")
            print(data[i, j])

if __name__ == "__main__":
    file_path = "processed_data/X_20250518_063031.npy"
    analyze_x_data(file_path) 