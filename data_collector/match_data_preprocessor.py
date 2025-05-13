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
            
            # 각 프레임의 이벤트에서 필요한 정보 추출
            frame_data = {
                'timestamp': timestamp,
                'positions': {}  # 참가자별 위치 정보
            }
            
            # 모든 이벤트에서 위치 정보 추출
            for event in events:
                # 이벤트에 참가자 정보가 있는 경우
                if 'participants' in event:
                    for participant in event['participants']:
                        if 'position' in event:
                            frame_data['positions'][participant] = {
                                'x': event['position']['x'],
                                'y': event['position']['y']
                            }
                
                # 이벤트 자체가 위치 정보를 포함하는 경우
                if 'position' in event:
                    if 'participantId' in event:
                        participant = event['participantId']
                        frame_data['positions'][participant] = {
                            'x': event['position']['x'],
                            'y': event['position']['y']
                        }
            
            processed_frames.append(frame_data)
        
        return processed_frames
    
    def create_time_series_data(self, processed_frames, target_participant=7):
        """시계열 데이터 생성 및 정렬"""
        # 0~600000 타임스탬프에 대한 데이터 포인트 생성 (1초 간격)
        timestamps = np.arange(0, 600001, 1000)
        position_data = np.zeros((len(timestamps), 2))  # x, y 좌표
        
        # 각 타임스탬프에 가장 가까운 프레임의 위치 정보 사용
        for i, ts in enumerate(timestamps):
            closest_frame = min(processed_frames, key=lambda x: abs(x['timestamp'] - ts))
            if target_participant in closest_frame['positions']:
                pos = closest_frame['positions'][target_participant]
                position_data[i] = [pos['x'], pos['y']]
            else:
                # 이전 프레임의 위치 정보를 사용
                if i > 0:
                    position_data[i] = position_data[i-1]
        
        return timestamps, position_data
    
    def normalize_positions(self, position_data):
        """위치 데이터 정규화 (0~1 범위로)"""
        # 맵 크기 (LoL 맵은 대략 15000 x 15000)
        map_size = 15000
        return position_data / map_size
    
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
            timestamps, position_data = self.create_time_series_data(processed_frames)
            
            # 위치 데이터 정규화
            normalized_data = self.normalize_positions(position_data)
            
            # 최종 위치 (600000 타임스탬프)를 레이블로 사용
            final_position = normalized_data[-1]
            
            # CNN 입력을 위한 데이터 형태로 변환
            # (600, 2) -> (1, 600, 2) 형태로 변환 (batch_size, sequence_length, features)
            input_data = normalized_data[:-1].reshape(1, -1, 2)
            
            processed_matches.append({
                'match_id': match_id,
                'input_data': input_data,
                'label': final_position
            })
        
        # 처리된 데이터 저장
        output_file = os.path.join(self.output_dir, f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy")
        np.save(output_file, processed_matches)
        print(f"처리된 데이터가 {output_file}에 저장되었습니다.")
        
        return processed_matches

def main():
    preprocessor = MatchDataPreprocessor()
    processed_data = preprocessor.process_all_matches()
    print(f"총 {len(processed_data)}개의 매치 데이터가 처리되었습니다.")

if __name__ == "__main__":
    main() 