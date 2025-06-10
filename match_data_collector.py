import requests
import time
import json
from datetime import datetime
from dotenv import load_dotenv
import os
import numpy as np
from match_data_preprocessor import MatchDataPreprocessor

# 프로젝트 루트 디렉토리 경로
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(ROOT_DIR, '.env')

# .env 파일 로드
load_dotenv(ENV_PATH)

class RateLimiter:
    def __init__(self):
        self.requests_20 = []   # 최근 1초 이내 요청 시간 기록
        self.requests_100 = []  # 최근 120초 이내 요청 시간 기록
    
    def wait_if_needed(self):
        current_time = time.time()
        
        # 1초당 20회 제한 체크
        self.requests_20 = [t for t in self.requests_20 if current_time - t < 1]
        if len(self.requests_20) >= 20:
            sleep_time = 1 - (current_time - self.requests_20[0])
            if sleep_time > 0:
                print(f"1초 제한 대기 중... {sleep_time:.2f}초")
                time.sleep(sleep_time)
        
        # 2분당 100회 제한 체크
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
        return [line.strip() for line in f if line.strip()]

def get_match_data(match_id, api_key, rate_limiter):
    """특정 match ID에 대해 metadata와 timeline 데이터를 동시에 요청하여 dict 반환"""
    base_url = "https://asia.api.riotgames.com/lol/match/v5/matches"
    headers = {"X-Riot-Token": api_key}
    
    # 최종 반환할 구조
    match_record = {
        "match_id": match_id,
        "metadata": None,
        "timeline": None
    }
    
    # 1) Metadata 요청
    try:
        rate_limiter.wait_if_needed()
        meta_url = f"{base_url}/{match_id}"
        resp = requests.get(meta_url, headers=headers)
        if resp.status_code == 200:
            match_record["metadata"] = resp.json()
        elif resp.status_code == 429:
            retry_after = int(resp.headers.get('Retry-After', 60))
            print(f"Rate limit (메타데이터) 초과. {retry_after}초 대기...")
            time.sleep(retry_after)
            return get_match_data(match_id, api_key, rate_limiter)
        else:
            print(f"메타데이터 요청 오류({resp.status_code}) for {match_id}")
            return None
    except Exception as e:
        print(f"메타데이터 요청 중 예외: {e}")
        return None
    
    # 2) Timeline 요청
    try:
        rate_limiter.wait_if_needed()
        tl_url = f"{base_url}/{match_id}/timeline"
        resp = requests.get(tl_url, headers=headers)
        if resp.status_code == 200:
            match_record["timeline"] = resp.json()
        elif resp.status_code == 429:
            retry_after = int(resp.headers.get('Retry-After', 60))
            print(f"Rate limit (타임라인) 초과. {retry_after}초 대기...")
            time.sleep(retry_after)
            return get_match_data(match_id, api_key, rate_limiter)
        else:
            print(f"타임라인 요청 오류({resp.status_code}) for {match_id}")
            return None
    except Exception as e:
        print(f"타임라인 요청 중 예외: {e}")
        return None
    
    return match_record

def main():
    # Riot API 키 (환경변수 또는 직접 입력)
    api_key = os.getenv("RIOT_API_KEY") or "RGAPI-..."
    if not api_key:
        raise ValueError("RIOT_API_KEY가 설정되어 있지 않습니다.")
    
    rate_limiter = RateLimiter()
    match_ids = read_match_ids("matchid_list.txt")
    match_ids = match_ids[:5000]  # 예: 처음 5000개만 처리
    
    preprocessor = MatchDataPreprocessor()
    collected = []
    
    for idx, match_id in enumerate(match_ids, start=1):
        print(f"Processing [{idx}/{len(match_ids)}]: {match_id}")
        record = get_match_data(match_id, api_key, rate_limiter)
        if record is not None:
            collected.append(record)
        
        # 100개가 모이면 바로 전처리 → 저장
        if len(collected) == 100:
            print(f"\n--- 100개 매치 수집 완료. 즉시 전처리 시작 ---")
            preprocessor.process_matches(collected)
            collected = []  # 초기화
    
    # 남은 매치가 있으면 처리
    if collected:
        print(f"\n--- 남은 {len(collected)}개 매치 전처리 시작 ---")
        preprocessor.process_matches(collected)
    
    print("모든 매치 수집 및 전처리 완료.")

if __name__ == "__main__":
    main()
