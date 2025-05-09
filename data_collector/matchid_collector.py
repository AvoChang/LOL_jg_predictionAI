import requests
import time
from datetime import datetime
import json

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

def read_puuid_list(filename):
    """puuid 리스트를 파일에서 읽어오는 함수"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def get_match_ids(puuid, api_key, rate_limiter):
    """특정 puuid의 match ID를 가져오는 함수"""
    url = f"https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count=100"
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
            return get_match_ids(puuid, api_key, rate_limiter)  # 재시도
        else:
            print(f"Error for puuid {puuid}: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

def main():
    # API 키 설정
    api_key = "my_api_key"
    
    # Rate limiter 초기화
    rate_limiter = RateLimiter()
    
    # puuid 리스트 파일 읽기
    puuid_list = read_puuid_list("puuid_list.txt")
    
    # match ID를 저장할 set
    all_match_ids = set()
    
    # 각 puuid에 대해 match ID 수집
    for i, puuid in enumerate(puuid_list):
        print(f"Processing puuid {i+1}/{len(puuid_list)}")
        match_ids = get_match_ids(puuid, api_key, rate_limiter)
        all_match_ids.update(match_ids)
    
    # 결과를 파일로 저장
    with open("matchid_list.txt", 'w', encoding='utf-8') as f:
        for match_id in sorted(all_match_ids):
            f.write(f"{match_id}\n")
    
    print(f"\n총 {len(all_match_ids)}개의 고유한 match ID가 수집되었습니다.")
    print("match ID 리스트가 matchid_list.txt 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()
