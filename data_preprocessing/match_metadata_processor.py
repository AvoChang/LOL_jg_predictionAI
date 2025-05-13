import json
import os

# 프로젝트 루트 디렉토리 경로 설정
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def extract_match_data(metadata_file):
    """match_metadata.txt에서 championId만 추출하는 함수"""
    file_path = os.path.join(ROOT_DIR, metadata_file)
    
    # metadata 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    
    # 추출된 데이터를 저장할 리스트
    extracted_data = []
    
    # 각 매치 데이터 처리
    for match_data in metadata_list:
        match_id = match_data['match_id']
        metadata = match_data['metadata']
        
        # championId만 리스트로 추출
        champion_ids = []
        for participant in metadata['info']['participants']:
            champion_ids.append(participant['championId'])
        
        # 추출된 데이터 저장
        extracted_data.append({
            'matchId': match_id,
            'championIds': champion_ids
        })
    
    return extracted_data

def save_extracted_data(data, output_file):
    """추출된 데이터를 파일로 저장하는 함수"""
    output_path = os.path.join(ROOT_DIR, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # 데이터 추출
    extracted_data = extract_match_data('match_metadata.txt')
    
    # 추출된 데이터 저장
    save_extracted_data(extracted_data, 'match_metadata_processed.txt')
    
    print(f"\n총 {len(extracted_data)}개의 매치 데이터가 처리되었습니다.")
    print("처리된 데이터가 match_metadata_processed.txt 파일에 저장되었습니다.")

if __name__ == "__main__":
    main() 