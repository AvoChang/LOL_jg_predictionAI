import json
import numpy as np
import os
from datetime import datetime

class MatchDataPreprocessor:
    def __init__(self):
        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.input_file = os.path.join(self.ROOT_DIR, "match_data.txt")
        self.output_dir = os.path.join(self.ROOT_DIR, "processed_data")
        
        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 특성 목록 정의
        self.features = [
            'health', 'healthMax', 'currentGold', 'totalDamageDone',
            'totalDamageDoneToChampions', 'jungleMinionsKilled',
            'level', 'minionsKilled', 'position_x', 'position_y',
            'totalGold'
        ]
    
    def load_match_data(self):
        """match_data.txt 파일에서 데이터를 로드"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_timeline_data(self, timeline_data):
        """타임라인 데이터에서 필요한 정보 추출"""
        frames = timeline_data['info']['frames']
        processed_frames = []
        
        for frame in frames:
            timestamp = frame['timestamp']
            events = frame['events']
            participants = frame.get('participantFrames', {})
            
            # 각 프레임의 이벤트에서 필요한 정보 추출
            frame_data = {
                'timestamp': timestamp,
                'participants': {}
            }
            
            # 각 참가자의 정보 추출
            for participant_id, participant_data in participants.items():
                participant_id = int(participant_id)
                champion_stats = participant_data.get('championStats', {})
                damage_stats = participant_data.get('damageStats', {})
                position = participant_data.get('position', {})
                
                frame_data['participants'][participant_id] = {
                    'health': champion_stats.get('health', 0),
                    'healthMax': champion_stats.get('healthMax', 0),
                    'currentGold': participant_data.get('currentGold', 0),
                    'totalDamageDone': damage_stats.get('totalDamageDone', 0),
                    'totalDamageDoneToChampions': damage_stats.get('totalDamageDoneToChampions', 0),
                    'jungleMinionsKilled': participant_data.get('jungleMinionsKilled', 0),
                    'level': participant_data.get('level', 1),
                    'minionsKilled': participant_data.get('minionsKilled', 0),
                    'position_x': position.get('x', 0),
                    'position_y': position.get('y', 0),
                    'totalGold': participant_data.get('totalGold', 0)
                }
            
            processed_frames.append(frame_data)
        
        return processed_frames
    
    def create_time_series_data(self, processed_frames):
        """시계열 데이터 생성 및 정렬"""
        # 0~600000 타임스탬프에 대한 데이터 포인트 생성 (1초 간격)
        timestamps = np.arange(0, 600001, 30000)
        
        # 3차원 배열 생성 (timestamps, participants, features)
        data = np.zeros((len(timestamps), 10, len(self.features)))
        
        # 각 타임스탬프에 가장 가까운 프레임의 정보 사용
        for i, ts in enumerate(timestamps):
            closest_frame = min(processed_frames, key=lambda x: abs(x['timestamp'] - ts))
            
            for participant_id in range(1, 11):  # 1부터 10까지의 참가자
                if participant_id in closest_frame['participants']:
                    participant_data = closest_frame['participants'][participant_id]
                    for j, feature in enumerate(self.features):
                        data[i, participant_id-1, j] = participant_data[feature]
                else:
                    # 이전 프레임의 데이터 사용
                    if i > 0:
                        data[i, participant_id-1] = data[i-1, participant_id-1]
        
        return timestamps, data
    
    def normalize_data(self, data):
        """데이터 정규화"""
        # 각 특성별로 정규화
        normalized_data = np.zeros_like(data)
        
        # 위치 데이터 정규화 (맵 크기 15000 x 15000)
        map_size = 15000
        normalized_data[:, :, 12:14] = data[:, :, 12:14] / map_size
        
        # 나머지 특성들 정규화
        for i in range(len(self.features)):
            if i not in [12, 13]:  # position_x, position_y는 이미 정규화됨
                feature_data = data[:, :, i]
                max_val = np.max(feature_data)
                if max_val > 0:
                    normalized_data[:, :, i] = feature_data / max_val
        
        return normalized_data
    
    def process_all_matches(self):
        """모든 매치 데이터 처리"""
        match_data = self.load_match_data()
        processed_matches = []
        
        for match in match_data:
            match_id = match['match_id']
            timeline = match['timeline']
            
            # 타임라인 데이터 처리
            processed_frames = self.extract_timeline_data(timeline)
            
            # 시계열 데이터 생성
            timestamps, data = self.create_time_series_data(processed_frames)
            
            # 데이터 정규화
            normalized_data = self.normalize_data(data)
            
            # 최종 위치 (600000 타임스탬프)를 레이블로 사용
            final_position = normalized_data[-1, 6, 12:14]  # participant 7의 최종 위치
            
            # CNN 입력을 위한 데이터 형태로 변환
            # (timestamps, participants, features) 형태 유지
            input_data = normalized_data[:-1]  # 마지막 타임스탬프 제외
            
            processed_matches.append({
                'match_id': match_id,
                'input_data': input_data,
                'label': final_position
            })
        
        # 처리된 데이터 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # npy 형식으로 저장
        npy_output_file = os.path.join(self.output_dir, f"processed_data_{timestamp}.npy")
        np.save(npy_output_file, processed_matches)
        print(f"처리된 데이터가 {npy_output_file}에 저장되었습니다.")
        
        # txt 형식으로 저장
        txt_output_file = os.path.join(self.output_dir, f"processed_data_{timestamp}.txt")
        with open(txt_output_file, 'w', encoding='utf-8') as f:
            for match in processed_matches:
                f.write(f"Match ID: {match['match_id']}\n")
                f.write("Input Data Shape: " + str(match['input_data'].shape) + "\n")
                f.write("Input Data:\n" + str(match['input_data']) + "\n")
                f.write("Label (Final Position): " + str(match['label']) + "\n")
                f.write("-" * 80 + "\n")
        print(f"처리된 데이터가 {txt_output_file}에 저장되었습니다.")
        
        # json 형식으로 저장
        json_output_file = os.path.join(self.output_dir, f"processed_data_{timestamp}.json")
        json_data = []
        for match in processed_matches:
            json_data.append({
                'match_id': match['match_id'],
                'input_data': match['input_data'].tolist(),
                'label': match['label'].tolist()
            })
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"처리된 데이터가 {json_output_file}에 저장되었습니다.")
        
        return processed_matches

def main():
    preprocessor = MatchDataPreprocessor()
    processed_data = preprocessor.process_all_matches()
    print(f"총 {len(processed_data)}개의 매치 데이터가 처리되었습니다.")

if __name__ == "__main__":
    main() 