import numpy as np
import os
from datetime import datetime

class MatchDataPreprocessor:
    def __init__(self):
        # 프로젝트 루트 및 출력 디렉토리 설정
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.ROOT_DIR, "processed_data")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 사용할 피처 목록 (챔피언 ID, 체력, 골드, 피해량 등)
        self.features = [
            'championId', 'currentHealth', 'currentGold', 'totalDamageDone',
            'totalDamageDoneToChampions', 'jungleMinionsKilled',
            'level', 'minionsKilled', 'totalGold', 'position_x', 'position_y'
        ]
    
    def extract_timeline_data(self, timeline_json, metadata_json):
        """
        한 매치의 timeline_json 과 metadata_json을 받아,
        각 프레임(frame)마다 10명 참가자의 피처(feature)들을 뽑아낸 뒤 리스트로 반환.
        """
        frames = timeline_json['metadata'] if False else None
        # 실제로 Riot 차원 구조: timeline_json['info']['frames']
        frames = timeline_json['info']['frames']
        
        processed_frames = []
        # metadata_json에서 participant별 championId를 미리 추출
        champion_ids = {}
        for i, part in enumerate(metadata_json['info']['participants']):
            champion_ids[i + 1] = part['championId']
        
        for frame in frames:
            timestamp = frame['timestamp']
            events = frame.get('events', [])
            participants = frame.get('participantFrames', {})
            
            frame_data = {'timestamp': timestamp, 'participants': {}}
            for pid_str, pdata in participants.items():
                pid = int(pid_str)  # 1~10
                champ_stats = pdata.get('championStats', {})
                dmg_stats = pdata.get('damageStats', {})
                pos = pdata.get('position', {})
                
                frame_data['participants'][pid] = {
                    'championId': champion_ids[pid],
                    'currentHealth': champ_stats.get('currentHealth', 0) / max(champ_stats.get('maxHealth', 1), 1),
                    'currentGold': pdata.get('currentGold', 0),
                    'totalDamageDone': dmg_stats.get('totalDamageDone', 0),
                    'totalDamageDoneToChampions': dmg_stats.get('totalDamageDoneToChampions', 0),
                    'jungleMinionsKilled': pdata.get('jungleMinionsKilled', 0),
                    'level': pdata.get('level', 1),
                    'minionsKilled': pdata.get('minionsKilled', 0),
                    'totalGold': pdata.get('totalGold', 0),
                    'position_x': pos.get('x', 0),
                    'position_y': pos.get('y', 0)
                }
            
            processed_frames.append(frame_data)
        
        return processed_frames
    
    def create_time_series_data(self, processed_frames):
        """
        processed_frames: [{'timestamp':..., 'participants':{pid: {features...}}}, ...]
        0~600000ms 구간을 60000ms(60초) 단위로 11포인트(0,60k,120k,...,600k)로 맞춰서
        각 참가자별 11개 타임스텝에 대한 피처를 채워 반환.
        """
        # 0부터 600000까지 60000 간격 → 11개 시점 (0,60k,120k,...,600k)
        timestamps = np.arange(0, 600001, 60000, dtype=np.int64)  # shape (11,)
        
        num_t = len(timestamps)
        num_p = 10
        num_f = len(self.features)
        
        # (11 타임스텝, 10명, 11개 피처) 형태
        data = np.zeros((num_t, num_p, num_f), dtype=np.float32)
        
        # 각 타임스텝(i)마다 closet_frame을 찾아서 값 채우기
        for i, ts in enumerate(timestamps):
            # timestamp 기준 가장 가까운 프레임
            closest = min(processed_frames, key=lambda x: abs(x['timestamp'] - ts))
            for pid in range(1, num_p + 1):
                if pid in closest['participants']:
                    pd = closest['participants'][pid]
                    for j, feat in enumerate(self.features):
                        data[i, pid-1, j] = float(pd[feat])
                else:
                    # 누락된 경우 바로 이전 타임스텝 복사
                    if i > 0:
                        data[i, pid-1, :] = data[i-1, pid-1, :]
        
        # (11, 10, 11) → (11, 10*11=110) 으로 reshape
        data_reshaped = data.reshape(num_t, num_p * num_f)
        return timestamps, data_reshaped
    
    def normalize_data(self, data):
        """
        data: (11, 110) 형태
        features 순서: 각 참가자마다 [championId, currentHealth, currentGold, ..., position_x, position_y]
        - championId(인덱스 0,11,22,...), currentHealth(1,12,...)은 이미 0~1 범위이므로 그대로 두고
        - position_x/y(마지막 두 피처)도 그대로 둔다.
        - 나머지(예: currentGold, totalDamageDone 등)는 열별로 max 값으로 나눠 정규화.
        """
        normalized = np.zeros_like(data, dtype=np.float32)
        num_t, feat_dim = data.shape
        num_f = len(self.features)  # 11
        num_p = 10
        
        # “건너뛸 인덱스” 집합: championId, currentHealth, position_x, position_y
        skip_idx = set()
        for p in range(num_p):
            base = p * num_f
            skip_idx.add(base + 0)  # championId
            skip_idx.add(base + 1)  # currentHealth
            skip_idx.add(base + 9)  # position_x
            skip_idx.add(base + 10) # position_y
        
        for col in range(feat_dim):
            col_vals = data[:, col]
            if col in skip_idx:
                normalized[:, col] = col_vals
            else:
                maxv = col_vals.max()
                if maxv > 0:
                    normalized[:, col] = col_vals / maxv
                else:
                    normalized[:, col] = col_vals
        
        return normalized
    
    def process_matches(self, match_records):
        """
        match_records: [
            {"match_id":..., "metadata":{...}, "timeline":{...}},
            ...
        ]
        한 번에 여러 매치를 받아서 각각 (11,110) 입력과 (2,) 라벨로 가공한 뒤,
        batch 단위로 .npy 파일로 저장.
        """
        batch_size = len(match_records)
        processed_list = []
        
        for idx, rec in enumerate(match_records):
            match_id = rec["match_id"]
            meta = rec["metadata"]
            tl   = rec["timeline"]
            
            # 1) timeline 처리 → processed_frames (리스트)
            frames = self.extract_timeline_data(tl, meta)
            
            # 2) (timestamps, data_shape=(11,110))
            timestamps, data_110 = self.create_time_series_data(frames)
            
            # 3) 정규화 → (11,110)
            normalized = self.normalize_data(data_110)
            
            # 4) 레이블 생성: 마지막 타임스텝(600k)의 7번 플레이어 위치 (0-based index=6)
            p_idx = 6
            x_idx = p_idx * len(self.features) + 9   # position_x
            y_idx = p_idx * len(self.features) + 10  # position_y
            label_xy = normalized[-1, [x_idx, y_idx]]  # (2,)
            
            # 5) 입력 데이터의 마지막 위치 정보는 0으로 씌워 버려서 “예측해야 할 정보”로 만들기
            input_copy = normalized.copy()
            input_copy[-1, [x_idx, y_idx]] = 0.0
            
            processed_list.append({
                "match_id": match_id,
                "input_data": input_copy,   # (11,110)
                "label": label_xy.astype(np.float32)
            })
        
        # 6) 배치 전체를 하나의 .npy 파일로 저장
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_filename = f"processed_data_{batch_size}matches_{timestamp_str}.npy"
        batch_path = os.path.join(self.output_dir, batch_filename)
        np.save(batch_path, np.array(processed_list, dtype=object))
        print(f">>> {batch_size}개 매치 처리 완료 및 저장: {batch_path}")
        return True
