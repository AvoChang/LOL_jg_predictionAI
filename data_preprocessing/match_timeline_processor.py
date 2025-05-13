import json
import os

# 프로젝트 루트 디렉토리 경로 설정
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def extract_timeline_data(timeline_file):
    """match_timeline.txt에서 15분까지의 데이터를 추출하는 함수"""
    file_path = os.path.join(ROOT_DIR, timeline_file)
    
    # timeline 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        timeline_list = json.load(f)
    
    # 추출된 데이터를 저장할 리스트
    extracted_data = []
    
    # 각 매치 데이터 처리
    for match_data in timeline_list:
        match_id = match_data['match_id']
        timeline = match_data['timeline']
        
        # 각 프레임(타임스탬프)별 데이터 처리
        for frame in timeline['info']['frames']:
            timestamp = frame['timestamp']
            
            # 15분(900000) 이후의 데이터는 건너뛰기
            if timestamp > 900000:
                continue
                
            frame_data = [timestamp]  # 첫 번째 값은 timestamp
            
            # 각 참가자별 데이터 처리
            for participant in frame['participantFrames'].values():
                # 기본 데이터 추출
                data = [
                    participant.get('damageStats', {}).get('damageDealtToObjectives', None),
                    participant.get('championStats', {}).get('abilityPower', None),
                    participant.get('championStats', {}).get('armor', None),
                    participant.get('championStats', {}).get('magicResist', None),
                    participant.get('championStats', {}).get('attackSpeed', None),
                    participant.get('championStats', {}).get('health', None),
                    participant.get('championStats', {}).get('healthMax', None),
                    participant.get('currentGold', None),
                    participant.get('damageStats', {}).get('totalDamageDone', None),
                    participant.get('damageStats', {}).get('totalDamageDoneToChampions', None),
                    participant.get('jungleMinionsKilled', None),
                    participant.get('level', None),
                    participant.get('minionsKilled', None),
                    participant.get('participantId', None),
                    participant.get('position', {}).get('x', None),
                    participant.get('position', {}).get('y', None),
                    participant.get('totalGold', None)
                ]
                frame_data.extend(data)
            
            extracted_data.append(frame_data)
    
    return extracted_data

def save_extracted_data(data, output_file):
    """추출된 데이터를 파일로 저장하는 함수"""
    output_path = os.path.join(ROOT_DIR, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # 데이터 추출
    extracted_data = extract_timeline_data('match_timeline.txt')
    
    # 추출된 데이터 저장
    save_extracted_data(extracted_data, 'match_timeline_processed.txt')
    
    print(f"\n총 {len(extracted_data)}개의 타임스탬프 데이터가 처리되었습니다.")
    print("처리된 데이터가 match_timeline_processed.txt 파일에 저장되었습니다.")

if __name__ == "__main__":
    main() 