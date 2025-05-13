import requests
import time
import json
from datetime import datetime
from dotenv import load_dotenv
import os

# 프로젝트 루트 디렉토리 경로 설정
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, '.env')

# .env 파일 로드
load_dotenv(ENV_PATH)

class RateLimiter:
    def __init__(self):
        self.requests_20 = []  # 1초당 20개 요청 제한
        self.requests_100 = []  # 2분당 100개 요청 제한
    
    def wait_if_needed(self):
        current_time = time.time()
        
        # 1초당 20개 요청 제한 체크
        self.requests_20 = [t for t in self.requests_20 if current_time - t < 1]
        if len(self.requests_20) >= 20:
            sleep_time = 1 - (current_time - self.requests_20[0])
            if sleep_time > 0:
                print(f"1초 제한 대기 중... {sleep_time:.2f}초")
                time.sleep(sleep_time)
        
        # 2분당 100개 요청 제한 체크
        self.requests_100 = [t for t in self.requests_100 if current_time - t < 120]
        if len(self.requests_100) >= 100:
            sleep_time = 120 - (current_time - self.requests_100[0])
            if sleep_time > 0:
                print(f"2분 제한 대기 중... {sleep_time:.2f}초")
                time.sleep(sleep_time)
        
        # 현재 요청 시간 기록
        self.requests_20.append(current_time)
        self.requests_100.append(current_time)

def read_match_ids(filename):
    """match ID 리스트를 파일에서 읽어오는 함수"""
    file_path = os.path.join(ROOT_DIR, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def get_match_data(match_id, api_key, rate_limiter):
    """특정 match ID의 metadata와 timeline 데이터를 동시에 가져오는 함수"""
    base_url = "https://asia.api.riotgames.com/lol/match/v5/matches"
    headers = {
        "X-Riot-Token": api_key
    }
    
    match_data = {
        "match_id": match_id,
        "metadata": None,
        "timeline": None
    }
    
    # Metadata 요청
    try:
        rate_limiter.wait_if_needed()
        metadata_url = f"{base_url}/{match_id}"
        metadata_response = requests.get(metadata_url, headers=headers)
        
        if metadata_response.status_code == 200:
            match_data["metadata"] = metadata_response.json()
        elif metadata_response.status_code == 429:
            retry_after = int(metadata_response.headers.get('Retry-After', 60))
            print(f"Rate limit exceeded. Waiting for {retry_after} seconds...")
            time.sleep(retry_after)
            return get_match_data(match_id, api_key, rate_limiter)
        else:
            print(f"Error for match metadata {match_id}: {metadata_response.status_code}")
            return None
    except Exception as e:
        print(f"Error occurred while fetching metadata: {e}")
        return None
    
    # Timeline 요청
    try:
        rate_limiter.wait_if_needed()
        timeline_url = f"{base_url}/{match_id}/timeline"
        timeline_response = requests.get(timeline_url, headers=headers)
        
        if timeline_response.status_code == 200:
            match_data["timeline"] = timeline_response.json()
        elif timeline_response.status_code == 429:
            retry_after = int(timeline_response.headers.get('Retry-After', 60))
            print(f"Rate limit exceeded. Waiting for {retry_after} seconds...")
            time.sleep(retry_after)
            return get_match_data(match_id, api_key, rate_limiter)
        else:
            print(f"Error for match timeline {match_id}: {timeline_response.status_code}")
            return None
    except Exception as e:
        print(f"Error occurred while fetching timeline: {e}")
        return None
    
    return match_data

def main():
    # API 키 설정
    api_key = os.getenv('RIOT_API_KEY')
    if not api_key:
        raise ValueError("RIOT_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
    
    # Rate limiter 초기화
    rate_limiter = RateLimiter()
    
    # match ID 리스트 읽기
    match_ids = read_match_ids("matchid_list.txt")
    
    # 최대 10개 경기만 처리
    match_ids = match_ids[:1]
    
    # match 데이터를 저장할 리스트
    match_data_list = []
    
    # 각 match ID에 대해 데이터 수집
    for i, match_id in enumerate(match_ids):
        print(f"Processing match {i+1}/{len(match_ids)}")
        match_data = get_match_data(match_id, api_key, rate_limiter)
        
        if match_data:
            match_data_list.append(match_data)
    
    # 결과를 파일로 저장
    output_path = os.path.join(ROOT_DIR, "match_data.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(match_data_list, f, ensure_ascii=False, indent=2)
    
    print(f"\n총 {len(match_data_list)}개의 match 데이터가 수집되었습니다.")
    print("데이터가 match_data.txt 파일에 저장되었습니다.")

if __name__ == "__main__":
    main() 