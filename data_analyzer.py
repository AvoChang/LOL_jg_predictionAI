import numpy as np
import json

def analyze_numpy_data(file_path):
    # 데이터 로드
    data = np.load(file_path, allow_pickle=True)
    
    print("=== 데이터 분석 결과 ===")
    print(f"데이터 타입: {type(data)}")
    print(f"데이터 형태: {data.shape}")
    
    # 데이터 내용 출력
    print("\n데이터 내용:")
    for i, item in enumerate(data):
        print(f"\n항목 {i}:")
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, np.ndarray):
                    print(f"{key}: {value.shape} 형태의 배열")
                    print(f"데이터 타입: {value.dtype}")
                    print(f"처음 5개 값: {value[:5]}")
                else:
                    print(f"{key}: {value}")
        else:
            print(item)

if __name__ == "__main__":
    file_path = "processed_data/processed_data_100matches_20250606_161028.npy"
    analyze_numpy_data(file_path) 