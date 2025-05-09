import requests
import os
from datetime import datetime

# API 키 설정
api_key = "my_api_key"

# 챌린저와 그랜드마스터 리그 엔드포인트
challenger_url = f"https://kr.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5?api_key={api_key}"
grandmaster_url = f"https://kr.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5?api_key={api_key}"

# API 요청 헤더
headers = {
    "X-Riot-Token": api_key
}

# puuid를 저장할 리스트
puuid_list = []

# 챌린저 리그 정보 가져오기
challenger_response = requests.get(challenger_url, headers=headers)
if challenger_response.status_code == 200:
    challenger_data = challenger_response.json()
    # entries 리스트에서 각 항목의 puuid 추출
    for entry in challenger_data['entries']:
        puuid_list.append(entry['puuid'])

# 그랜드마스터 리그 정보 가져오기
grandmaster_response = requests.get(grandmaster_url, headers=headers)
if grandmaster_response.status_code == 200:
    grandmaster_data = grandmaster_response.json()
    # entries 리스트에서 각 항목의 puuid 추출
    for entry in grandmaster_data['entries']:
        puuid_list.append(entry['puuid'])

# 현재 시간을 파일명에 포함
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"puuid_list_{current_time}.txt"

# 파일에 puuid 리스트 저장
with open(filename, 'w', encoding='utf-8') as f:
    for puuid in puuid_list:
        f.write(f"{puuid}\n")

print(f"총 {len(puuid_list)}개의 puuid가 수집되었습니다.")
print(f"puuid 리스트가 {filename} 파일에 저장되었습니다.")