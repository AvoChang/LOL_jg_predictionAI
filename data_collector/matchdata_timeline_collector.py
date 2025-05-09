import requests
import time
import json
from datetime import datetime

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
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def get_match_timeline(match_id, api_key, rate_limiter):
    """특정 match ID의 timeline 데이터를 가져오는 함수"""
    url = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    headers = {
        "X-Riot-Token": api_key
    }
    
    try:
        rate_limiter.wait_if_needed()  # 요청 제한 체크
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:  # Rate limit exceeded
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limit exceeded. Waiting for {retry_after} seconds...")
            time.sleep(retry_after)
            return get_match_timeline(match_id, api_key, rate_limiter)  # 재시도
        else:
            print(f"Error for match {match_id}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def main():
    # API 키 설정
    api_key = "my_api_key"
    
    # Rate limiter 초기화
    rate_limiter = RateLimiter()
    
    # match ID 리스트 읽기
    match_ids = read_match_ids("matchid_list.txt")
    
    # 최대 1000개 경기만 처리
    match_ids = match_ids[:5000]
    
    # timeline 데이터를 저장할 리스트
    timeline_data = []
    
    # 각 match ID에 대해 timeline 데이터 수집
    for i, match_id in enumerate(match_ids):
        print(f"Processing match {i+1}/{len(match_ids)}")
        timeline = get_match_timeline(match_id, api_key, rate_limiter)
        
        if timeline:
            timeline_data.append({
                "match_id": match_id,
                "timeline": timeline
            })
    
    # 결과를 파일로 저장
    with open("matchdata_timeline.txt", 'w', encoding='utf-8') as f:
        json.dump(timeline_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n총 {len(timeline_data)}개의 match timeline 데이터가 수집되었습니다.")
    print("데이터가 matchdata_timeline.txt 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()