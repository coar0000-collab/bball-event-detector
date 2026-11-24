# bball-event-detector (garbage-time-tracker)

농구 중계 영상에서 **스코어버그(점수판)를 읽어 점수 변화 이벤트를 뽑고**, **코트 라인을 기준으로 슛 위치가 2점인지 3점인지(코너·45도·탑)** 판정하는 도구입니다. GARBAGE TIME PROJECT의 자동 기록 파이프라인 1단계입니다.

지금 되는 것

| 기능 | 명령 | 결과 |
| --- | --- | --- |
| 스코어버그 점수 인식 | `scorebug-setup` → `scorebug-learn` → `scorebug-read` | `out/score_events.csv` (언제, 어느 팀이 +1/+2/+3) |
| 코트 라인 캘리브레이션 | `calibrate` | `configs/calib.json` (화면 ↔ 코트 좌표 변환) |
| 팬·줌 카메라 추적 | `track-court` | `out/court_track.npz` (프레임별 코트 변환) |
| 라인 확인 | `overlay` | 영상 위에 코트 라인을 겹친 이미지 |
| 슛 위치 기록 + 2점/3점 판정 | `click-shots` | `out/shots.csv` (시각, 팀, 등번호, 성공/실패, 2점/3점, 구역, 라인까지 거리) |
| 기록지 점검 | `check-records` | 기록지 엑셀의 이상한 값 목록 |
| 설치 점검 | `demo` | 합성 중계로 전체 흐름을 돌린 결과와 걸린 시간 |
| 기록지와 비교 | `compare` | `out/compare.xlsx` (팀·선수별 기록지 vs 자동, 3점 구역 제안) + 구역을 채운 기록지 사본 |
| 코트 점수 검증 | `score-check` | 슛 위치(코트)로 정한 자유투/2점/3점이 스코어버그 +1/+2/+3 과 맞는 비율 |
| **영상 자동 분석 데모** | `analyze` | 영상 하나 → 기록지 엑셀(D3 [기록입력] 형식) + 평가지표 + 보고서 HTML |
| 정확도 벤치 | `bench` | 빠른 팬·줌, 리플레이, 방송 로고, 밝은 조명, 라인 근처 슛을 넣은 합성 중계로 점수 인식·코트 추적·코트 점수 정확도 측정 |

아직 안 되는 것: 슛을 사람이 클릭하지 않아도 자동으로 찾는 기능(공·림·선수 검출 모델)은 다음 단계입니다. `click-shots`로 쌓은 기록이 그 모델의 학습·평가 데이터가 됩니다.

## 영상 자동 분석 (`gtt analyze`)

영상을 넣으면 점수판 읽기 → 코트 추적 → (슛 기록) → **기록지 엑셀 + 평가지표 + 보고서**까지 한 번에 만듭니다.

```bash
gtt analyze --demo --open                     # 영상 없이: 합성 중계로 전체 흐름 (30초 안팎)
gtt analyze --video videos/game1.mp4 --open   # 실제 영상
```

Windows에서는 `analyze_video.bat` 위에 영상 파일을 끌어다 놓으면 됩니다(더블클릭하면 데모).

결과 폴더(`out/analysis_<영상 이름>/`)

| 파일 | 내용 |
| --- | --- |
| `report.html` | 최종 스코어, 점수 흐름 그래프, 기록지(팀·선수), 평가지표, 슛 위치, 코트 라인 추적 확인 그림, 득점 기록 |
| `기록지_자동.xlsx` | [기록입력]은 D3 기록지와 같은 열 순서(복사해 붙이기 가능), [팀 기록] [경기목록] [평가지표] [점수 흐름] [슛 기록] [구역별 슛] [설명] |
| `score_events.csv`, `court_track.npz`, `score_check.csv`, `summary.json` | 중간 결과 |

**자동으로 채우는 것과 아직인 것.** 초록 칸은 영상에서 자동, 회색 빈칸은 아직 자동 감지가 없는 항목입니다(0이 아니라 빈칸).

| 입력 | 채워지는 기록 |
| --- | --- |
| 영상만 | 팀 득점·득점 시각(점수판), 3점 성공(+3)·자유투 성공(+1) 횟수. +2는 2점인지 자유투 2개인지 몰라 2점 성공은 비워 둠 |
| 영상 + `--shots`(click-shots 결과) | 위 + 슛 시도·성공, 2점/3점(코트 라인), 3점 구역(코너·45도·탑), 등번호별 선수 기록, 코트로 정한 점수 vs 점수판 일치율 |
| 아직 없음 | 리바운드, 어시스트, 스틸, 블락, 턴오버, 파울 |

**구역별 슛.** 슛 위치를 카메라 화면이 아니라 코트 규격 좌표(m)로 바꾼 뒤 세부 구역 12개(림 근처, 페인트, 미드 베이스라인·45도 좌/우, 미드 정면, 코너·45도 좌/우, 탑)와 거리대 6개로 나눠 팀별 시도·성공을 셉니다(`gtt/zones.py`). 슛마다 코트 x·y, 림까지 거리, 각도도 [슛 기록]에 남습니다.

**코트로 못 정한 슛은 점수판으로 보정 (기본).** 라인 근처(사람 판정 없음)이거나 발이 가려져 코트 좌표가 없는 **성공** 슛은 그 순간 점수판 +2/+3으로 2점/3점을 정하고, 위치(코너·45도·탑)는 좌표의 방향으로 붙입니다. 실패 슛은 점수판이 안 바뀌어 '판정 불가'로 남고, 코트가 확실한데 점수판과 다르면 '코트≠점수판'으로 표시합니다. 보고서의 '코트로 정한 점수 = 점수판' 일치율은 보정 전 코트 단독 값이라 보정이 지표를 부풀리지 않습니다. 끄려면 `--court-only`.

**방송 프로필.** 중계 채널·체육관마다 한 번만 준비합니다(`--profile`, 기본 `configs/`): `scorebug.json`(scorebug-setup), `digits.npz`(scorebug-learn), `calib*.json`(calibrate, 선택). 같은 중계의 다른 경기는 바로 `analyze`만 하면 됩니다.

**선수 이름.** `--roster roster.csv`(열 `team,no,name`, team은 home/away 또는 팀 이름)를 주면 등번호 대신 이름이 들어갑니다. 실명이 든 명단은 깃에 올리지 마세요.

**평가지표 수식 바꾸기.** 수식은 코드가 아니라 `gtt/metrics_default.json`에 있습니다. 복사해서 고친 뒤 `--metrics 내수식.json`으로 주면 됩니다.

```json
{"metrics": [
  {"key": "ABEi", "name": "ABEi", "formula": "PTS + a*REB + b*AST + c*STL + d*BLK - e*TOV - f*PF",
   "params": {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0, "f": 1.0}, "format": "{:.1f}"}
]}
```

쓸 수 있는 기록: `PTS FGM FGA P2M P2A P3M P3A FTM FTA ORB DRB REB AST STL BLK TOV PF`, 사칙연산·괄호·`min`/`max`. 기본 파일에는 ABEi(가중치 a~f는 확정 전이라 임시로 1), Game Score, TS%, eFG%가 들어 있습니다. 수식에 아직 못 재는 기록이 들어 있으면 그 지표는 '계산 불가'로 두고, 참고용으로 그 기록을 0으로 둔 부분값과 빠진 기록 목록을 같이 보여 줍니다.

**데모의 정직한 범위.** `--demo`는 합성 중계입니다. 점수판 읽기·코트 추적·점수 판정은 실제 명령 그대로 돌리지만, 사람이 한 번 하는 준비(점수판 영역·숫자 등록, 코트 점 찍기)와 슛 위치 클릭은 합성 정답으로 대신합니다. 보고서 맨 위에 이 내용이 표시됩니다.

## 데스크톱 설치

Python 3.10 이상이 필요합니다. (Windows는 [python.org](https://www.python.org/downloads/)에서 설치할 때 "Add python.exe to PATH" 체크)

```bash
git clone https://github.com/coar0000-collab/bball-event-detector.git
cd bball-event-detector
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
pytest            # 테스트가 모두 통과하면 설치 완료
gtt demo          # 영상 없이 합성 중계 60초로 전체 흐름 점검 (1분 이내)
```

`gtt demo`는 팬·줌하는 합성 중계 영상을 만들어 `scorebug-read` → `track-court` → `overlay` → `shot`을 실제 명령과 똑같이 돌리고, 정답과 비교한 결과와 걸린 시간(40분 경기 예상)을 보여 줍니다. 창에서 사람이 하던 단계(스코어버그 영역·숫자 등록, 코트 점 찍기)는 정답 좌표로 대신합니다. `out/demo/overlay_grid.png`에서 코트 라인이 맞게 겹쳤는지 볼 수 있습니다.

영상 파일은 `videos/` 폴더에 두면 됩니다(깃에는 올라가지 않음). 유튜브 중계를 받을 때는 [yt-dlp](https://github.com/yt-dlp/yt-dlp)를 쓸 수 있습니다. **B-LAB 등 중계사 영상을 학습·서비스에 쓰려면 채널의 허락이 필요합니다.**

```bash
pip install yt-dlp
yt-dlp -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]" -o videos/game1.mp4 "<유튜브 주소>"
```

## 사용 순서

아래 예시는 `videos/game1.mp4` 기준입니다. 시간(`--t`)은 초 단위입니다.

### 1. 스코어버그 점수 인식

```bash
# (1) 점수 숫자 영역 지정: 창이 뜨면 왼쪽 팀 점수 숫자만 드래그 → Enter, 오른쪽 팀도 같은 방법
gtt scorebug-setup --video videos/game1.mp4 --t 900 --home 고려대 --away 연세대

# (2) 숫자 글꼴 등록: 화면에 나온 점수를 콘솔에 입력 (예: 47 52). 0~9가 다 나올 때까지
gtt scorebug-learn --video videos/game1.mp4 --n 12

# (3) 전체 경기 읽기 → 점수 변화 이벤트 CSV
gtt scorebug-read --video videos/game1.mp4 --out out/game1_score_events.csv
```

결과 CSV의 `flag`가 `ok`면 정상(+1/+2/+3), `check`나 `correction`이면 판독 오류나 점수 정정일 수 있으니 확인합니다. 기록원이 점수를 고친 흔적은 30초 안의 앞뒤 변화와 묶어 정리합니다: 잠깐 내렸다가 되돌린 증가는 `revert`(득점 아님), 넣었다가 지운 득점은 `reverted`, +3 을 넣었다가 -1 로 고친 것은 +2 `ok`(`note`에 표시)와 `merged`가 됩니다. `ok`만 득점으로 셉니다. `t_sec`는 스코어버그가 바뀐 시각이라 실제 슛보다 보통 1~5초 늦습니다. 엑셀에서 바로 열 수 있게 UTF-8(BOM)으로 저장합니다.

### 2. 코트 라인 캘리브레이션

```bash
gtt calibrate --video videos/game1.mp4 --t 900 --refine
```

- 오른쪽 `court reference` 창의 **빨간 점**이 지금 찍을 지점입니다. 영상 창에서 같은 지점을 클릭하세요.
- 안 보이는 지점은 `n`으로 건너뜁니다. 4개 이상 찍고 `f`를 누르면 노란 코트 라인이 겹쳐 보입니다.
- 노란 선이 실제 라인과 맞으면 `s`로 저장합니다. 베이스라인 점만 찍는 것처럼 한 직선 위의 점만 고르면 안 됩니다.
- 기본 규격은 FIBA(3점 6.75m, 코너 6.6m)입니다. 체육관 라인이 다르면 `gtt/court.py`의 `CourtSpec`에 규격을 추가하세요.

### 3. 카메라를 따라가며 코트 변환 계산

```bash
gtt track-court --video videos/game1.mp4 --calib configs/calib.json --step 0.2
gtt overlay --video videos/game1.mp4 --t 1500 --track out/court_track.npz --out out/check_1500.png
```

**양쪽 골대를 미리 맞춰 두세요.** 중계 카메라가 하프코트를 휙 가로지르면 반대쪽 골대 화면이 좌우만 바뀐 비슷한 모양이라, 기준이 없으면 다시 찾기가 어렵습니다. 왼쪽·오른쪽 골대가 잘 보이는 시각에서 각각 `calibrate`를 해 두고 `--calib`을 여러 번 주면, 첫 번째에서 출발하고 나머지는 놓쳤을 때 다시 찾는 기준으로 씁니다.

```bash
gtt calibrate --video videos/game1.mp4 --t 120 --refine --out configs/calib_right.json
gtt calibrate --video videos/game1.mp4 --t 15 --refine --out configs/calib_left.json
gtt track-court --video videos/game1.mp4 --calib configs/calib_right.json --calib configs/calib_left.json --step 0.2
```

빠른 패닝 때는 샘플 사이 프레임까지 이어서 카메라 이동을 계산합니다. 다음 경우에는 좌표를 내지 않습니다(틀린 좌표보다 빈칸이 낫다):

- 화면이 흐린 빠른 패닝 중 라인이 확실히 맞지 않을 때
- 라인 일치율이 0.65(`--min-score`)보다 낮을 때
- 화면에 보이는 코트 라인 줄(베이스라인·키 세 변·코너·아크 …) 가운데 실제 선과 겹치는 줄이 60% 미만이거나, 맞은 줄이 모두 평행한 가로선뿐일 때 (선 방향으로 미끄러져도 똑같이 맞아 위치가 정해지지 않음)
- 캘리브레이션 화면과 비교해 코트가 찌그러졌을 때 (키 투영 넓이가 10% 미만, 또는 화면 가운데의 코트가 훨씬 납작함). 이때는 화면 이동으로 예측한 위치를 들고 가다가 라인이 다시 맞으면 이어 갑니다.

좌표가 빈 구간이 4초(`--bridge`) 이하이고 앞뒤가 정상이면, 앞에서 화면 이동만으로 따라간 위치와 뒤에서 거꾸로 따라온 위치가 3px 안에서 서로 맞는 샘플만 채웁니다(상태 `bridged`). 앞에서 따라간 길이 뒤 정상 화면과 만나지 않으면 하나도 채우지 않습니다.

`--start`가 캘리브레이션 시각과 달라도 캘리브레이션 화면을 기준으로 시작 화면의 위치를 다시 찾습니다. 득점 장면에 코트가 비어 있으면 그 시각으로 `calibrate`를 한 번 더 하고 `--calib`에 추가해 다시 돌리면 됩니다.

`overlay` 이미지에서 노란 선이 라인에 겹쳐 있으면 추적이 잘 된 것입니다. 리플레이·클로즈업 뒤에 비슷한 화면으로 돌아오면 자동으로 다시 맞추고, 못 맞추면 그 구간은 코트 좌표가 비어 있습니다(그때는 그 시각으로 `calibrate`를 다시 하고 `track-court --start`로 이어서 실행).

### 4. 슛 위치 기록과 2점/3점 판정

```bash
gtt click-shots --video videos/game1.mp4 --track out/court_track.npz --out out/game1_shots.csv
```

| 키 | 동작 |
| --- | --- |
| space | 재생/멈춤 |
| a / d, j / l | ±1초, ±5초 |
| , / . | 한 프레임 앞뒤 |
| 클릭 | 릴리스 순간 슈터 두 발 가운데 → 화면에 2점/3점, 라인까지 거리 표시 (오른쪽 위에 4배 확대 보기) |
| Shift+클릭 | 같은 발을 다른 프레임에서 한 번 더 찍어 평균 (NEAR LINE 일 때) |
| f / r | 자유투 표시 / 리플레이 화면에서 찍은 슛 표시 |
| m / x | 성공 / 실패로 저장 |
| t | 홈/원정 전환 |
| z | 마지막 기록 삭제 |
| q | 종료 |

라인에서 30cm 안쪽이거나, 그 자리에서 발 위치를 3px 잘못 찍었을 때 흔들리는 거리(`margin_sigma_m`)의 2배 안쪽이면 `NEAR LINE!`이 뜨고 CSV의 `near_line`이 `True`가 됩니다. 카메라에서 먼 코너는 1px 이 코트에서 5cm 넘게 되므로 더 넓게 잡힙니다. 이 슛들은 사람이 한 번 더 보는 것을 권합니다.

슛을 저장하기 전에 숫자키로 슈터 등번호를 치면 같이 저장되고, 이 번호로 기록지 선수와 연결됩니다. `t`(홈/원정)는 스코어버그 **왼쪽/오른쪽** 팀이라는 뜻이며, 기록지의 홈/원정과 다르더라도 `compare`가 최종 스코어로 알아서 맞춥니다.

슛 하나만 빠르게 판정하려면:

```bash
gtt shot --track out/court_track.npz --t 1500 --u 812 --v 455
```

### 5. 기록지 엑셀과 비교

기록지 엑셀(`D3 기록지_1R-11R.xlsx` 형식)은 개인 기록이 들어 있으니 `data/` 폴더처럼 깃에 올라가지 않는 곳에 두세요(`.gitignore`에 `*.xlsx` 포함).

```bash
# 기준 데이터 점검: 득점 합계 불일치, 리바운드 열 뒤바뀜, 성공 > 시도 등
gtt check-records --records "data/D3 기록지_1R-11R.xlsx"

# 하루치 중계(여러 경기)를 읽은 점수 이벤트와 슛 기록을 경기별로 비교
gtt compare --records "data/D3 기록지_1R-11R.xlsx" --game D101,D102,D103 \
    --score-events out/day1_score_events.csv --shots out/day1_shots.csv \
    --out out/compare.xlsx --apply-zones "out/D3 기록지_구역채움.xlsx"
```

- 하루 중계 하나에 여러 경기가 들어 있어도 됩니다. 점수가 0:0 근처로 돌아가는 지점에서 경기를 나누고, 기록지 최종 스코어와 가장 가까운 구간을 그 경기로 고릅니다. 스코어버그 왼쪽 팀이 기록지의 홈인지 원정인지도 스코어로 정합니다(`--left-team`으로 직접 지정 가능).
- 보고서 엑셀 시트: **팀 비교**(득점, 스코어버그 +1/+2/+3 횟수, 슛 기록의 2점/3점/구역별 시도·성공), **선수 비교**(등번호를 입력한 슛만), **3점 구역 제안**. 차이가 있는 줄은 빨간색입니다.
- **3점 구역 제안**: 기록지 [3점슛_구역체크]의 3점 시도(공식 문자중계 순서)와 영상에서 클릭한 3점 시도를 팀별로 짝지어 코너/45도/탑을 제안합니다. 결과(성공/실패), 그 순간 스코어버그 점수와 '직전 스코어', 등번호가 맞을수록 확실하게 짝짓습니다.
- `--apply-zones`를 주면 상태가 "일치"인 제안만 기록지 **사본**의 구역 칸에 채웁니다. 원본은 건드리지 않고, 이미 구역이 있는 칸은 덮어쓰지 않으며, 드롭다운·수식은 그대로 둡니다.
- 스코어버그 +2 횟수에는 자유투 2개가 한 번에 반영된 경우가 섞일 수 있어, 2점 성공 수와 정확히 같지 않을 수 있습니다. 득점 합계와 +3 횟수가 가장 믿을 만한 비교 지표입니다.

## 코드 구성

| 파일 | 내용 |
| --- | --- |
| `gtt/court.py` | 코트 규격(FIBA/NBA), 랜드마크, 2점/3점·구역 판정 |
| `gtt/homography.py` | 화면↔코트 변환, 흰 라인 자동 정합, 팬·줌 카메라 추적 |
| `gtt/scorebug.py` | 스코어버그 숫자 분리·템플릿 인식, 점수 변화 이벤트 검출 |
| `gtt/records.py` | D3 기록지 엑셀 읽기와 이상값 점검 |
| `gtt/compare.py` | 경기 구간 찾기, 팀·선수 비교, 3점 구역 짝짓기, 보고서·구역 채우기 |
| `gtt/cli.py` | 명령줄 도구 |
| `gtt/demo.py` | `gtt demo`: 합성 중계로 전체 흐름 점검 |
| `gtt/bench.py` | `gtt bench`: 어려운 장면을 넣은 합성 중계로 정확도 측정 (같은 seed 면 같은 영상) |
| `gtt/analyze.py` | `gtt analyze`: 영상 → 기록지 엑셀 + 평가지표 + 보고서 HTML, `--demo` |
| `gtt/resolve.py` | 코트로 못 정한 성공 슛만 점수판으로 2점/3점 보정 (하이브리드) |
| `gtt/zones.py` | 코트 좌표 → 세부 구역·거리대, 구역별 시도/성공 |
| `gtt/boxscore.py` | 점수판 이벤트 + 슛 기록 → 팀·선수 기록 (잰 기록과 못 잰 기록 구분) |
| `gtt/metrics.py`, `gtt/metrics_default.json` | 평가지표 수식(설정 파일) 해석·계산 |
| `gtt/scorecheck.py` | 코트로 정한 점수 vs 스코어버그 점수 변화 짝짓기·일치율 |
| `tests/` | 합성 영상으로 만든 테스트 |

## 다음 단계

1. 실제 중계 영상 1경기로 `scorebug-read` 결과를 기록지와 비교해 정확도 확인
2. `click-shots`로 3경기 정도 슛 위치를 쌓아, 공·림 검출 모델(YOLO 계열) 학습 데이터로 사용
3. 스코어버그 점수 변화 시각 근처에서 공이 림을 통과하는 장면을 자동으로 찾아 슛 위치를 자동 제안
4. 스코어버그 점수 변화로 찾은 성공 슛 시각을 click-shots에 미리 띄워, 클릭할 장면으로 바로 이동
