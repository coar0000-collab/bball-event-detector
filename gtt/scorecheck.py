"""코트만으로 계산한 득점이 스코어버그(정답)와 얼마나 맞는지 잰다.

목표 지표: 스코어버그를 보지 않고 슛 위치(코트 라인·거리)로 정한 자유투/2점/3점이
스코어버그의 +1/+2/+3 과 일치하는 비율. 스코어버그 점수 변화는 정답으로만 쓴다.

짝짓기: 팀마다 점수 변화와 '성공' 슛을 시간 순서를 지키며 짝짓는다. 슛은 변화보다 max_lag 초 안쪽으로 앞서거나
lead 초 안쪽으로 뒤따라야 한다. 실제 중계에서 점수판이 16초 늦게 바뀐 적이 있어 max_lag 기본값은 20초이고, 순서를 지키므로
늦은 변화가 다음 슛을 가로채지 않는다. 스코어버그가 자유투 2개를 한 번에 +2 로 올리면 자유투 2개와 짝짓는다. 리플레이에서 찍은 슛(replay=True)은 득점보다 늦게 나오므로 replay_after 초 뒤까지 찾고,
본 화면에서 찍은 슛이 있으면 그쪽을 먼저 쓴다.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


def shot_points(row: dict) -> int | None:
    """슛 한 줄의 코트 기반 점수. 자유투로 표시됐으면 1."""
    if str(row.get("zone", "")).upper() == "FT" or str(row.get("kind", "")).upper() == "FT":
        return 1
    try:
        return int(float(row.get("points") or 0)) or None
    except ValueError:
        return None


def _is_replay(row: dict) -> bool:
    return str(row.get("replay", "")).lower() in ("true", "1")


def _made(row: dict) -> bool:
    return str(row.get("result", "")).upper() in ("MADE", "성공", "1", "TRUE")


@dataclass
class Pair:
    event: dict
    shots: list[dict]

    @property
    def court_points(self) -> int | None:
        if not self.shots:
            return None
        pts = [shot_points(s) for s in self.shots]
        return None if any(p is None for p in pts) else sum(pts)

    @property
    def bug_points(self) -> int:
        return int(self.event["delta"])

    @property
    def agree(self) -> bool | None:
        return None if self.court_points is None else self.court_points == self.bug_points


@dataclass
class Result:
    pairs: list[Pair] = field(default_factory=list)
    unmatched_shots: list[dict] = field(default_factory=list)  # 성공으로 적었는데 점수 변화가 없음

    def summary(self) -> dict:
        matched = [p for p in self.pairs if p.shots]
        judged = [p for p in matched if p.court_points is not None]  # 코트 좌표가 있어 점수를 낸 것
        agree = [p for p in judged if p.agree]
        conf: dict[str, int] = {}
        for p in judged:
            k = f"코트 {p.court_points} / 점수판 {p.bug_points}"
            conf[k] = conf.get(k, 0) + 1
        near = [p for p in judged if any(str(s.get("near_line")).lower() == "true" for s in p.shots)]
        totals = {}
        for team in ("home", "away"):
            bug = sum(p.bug_points for p in self.pairs if p.event["team"] == team)
            court = sum(shot_points(s) or 0 for p in matched if p.event["team"] == team for s in p.shots)
            court += sum(shot_points(s) or 0 for s in self.unmatched_shots if s.get("team") == team)
            totals[team] = {"점수판": bug, "코트": court}
        return {
            "점수판 변화": len(self.pairs),
            "짝지은 변화": len(matched),
            "코트 좌표 없음": len(matched) - len(judged),
            "일치": len(agree),
            "일치율(짝지은 것 중)": round(len(agree) / len(judged), 3) if judged else None,
            "일치율(전체 변화 중)": round(len(agree) / len(self.pairs), 3) if self.pairs else None,
            "슛 없이 바뀐 점수": len(self.pairs) - len(matched),
            "점수 변화 없는 성공 슛": len(self.unmatched_shots),
            "라인 근처 슛이 낀 변화": len(near),
            "라인 근처에서 틀림": sum(1 for p in near if not p.agree),
            # 라인에서 충분히 떨어져 코트만으로 확실히 정한 것. 여기서 틀리면 추적·찍기·짝짓기 문제다.
            "확실한 판정": len(judged) - len(near),
            "확실한 판정 일치": sum(1 for p in judged if p not in near and p.agree),
            "점수 비교": conf,
            "팀 합계": totals,
        }

    def rows(self) -> list[dict]:
        out = []
        for p in self.pairs:
            s0 = p.shots[0] if p.shots else {}
            out.append({
                "점수판 시각": p.event.get("time") or p.event["t_sec"], "팀": p.event["team"], "점수판": p.bug_points,
                "코트": p.court_points if p.court_points is not None else "", "일치": {True: "O", False: "X", None: "-"}[p.agree],
                "슛 시각": ", ".join(str(s.get("time") or s.get("t_sec")) for s in p.shots),
                "구역": ", ".join(str(s.get("zone_kr", "")) for s in p.shots),
                "3점 라인까지(m)": s0.get("margin_m", ""), "라인 근처": s0.get("near_line", ""),
                "자유투 자리": s0.get("ft_spot", ""),
            })
        for s in self.unmatched_shots:
            out.append({"점수판 시각": "", "팀": s.get("team"), "점수판": "", "코트": shot_points(s), "일치": "-",
                        "슛 시각": s.get("time") or s.get("t_sec"), "구역": s.get("zone_kr", ""),
                        "3점 라인까지(m)": s.get("margin_m", ""), "라인 근처": s.get("near_line", ""), "자유투 자리": s.get("ft_spot", "")})
        return out


def _align(E: list[dict], S: list[dict], max_lag: float, lead: float, skip: float = 1.0) -> list[tuple[int, list[int]]]:
    """한 팀의 점수 변화 E 와 성공 슛 S(둘 다 시간순)를 순서를 지키며 짝짓는다 (편집 거리 같은 동적 계획법).

    점수판은 득점 순서대로 바뀌므로, 늦게 반영된 변화(예: 16초 지연)가 다음 슛을 가로채지 않는다.
    비용: 짝이 없는 변화·슛은 skip, 짝은 지연(초)/max_lag 의 절반. 코트 점수는 짝짓기에 쓰지 않는다(지표가 공정하도록).
    +2 는 자유투 2개와도 짝지을 수 있다. 반환: [(E 번호, [S 번호...])]
    """
    n, m = len(E), len(S)
    te = [float(e["t_sec"]) for e in E]
    ts = [float(x["t_sec"]) for x in S]
    ok = lambda a, b: te[a] - max_lag <= ts[b] <= te[a] + lead
    lag = lambda a, b: 0.5 * max(0.0, te[a] - ts[b]) / max_lag
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for a in range(n + 1):
        for b in range(m + 1):
            c = cost[a][b]
            if c == INF:
                continue
            steps = []
            if a < n:
                steps.append((a + 1, b, c + skip, ()))
            if b < m:
                steps.append((a, b + 1, c + skip, None))
            if a < n and b < m and ok(a, b):
                steps.append((a + 1, b + 1, c + lag(a, b), (b,)))
            if (a < n and b + 1 < m and int(E[a]["delta"]) == 2 and ok(a, b) and ok(a, b + 1)
                    and shot_points(S[b]) == 1 and shot_points(S[b + 1]) == 1):
                steps.append((a + 1, b + 2, c + lag(a, b + 1), (b, b + 1)))
            for a2, b2, c2, used in steps:
                if c2 < cost[a2][b2] - 1e-12:
                    cost[a2][b2] = c2
                    back[a2][b2] = (a, b, used)
    out = []
    a, b = n, m
    while (a, b) != (0, 0):
        pa, pb, used = back[a][b]
        if used is not None:  # 변화 pa 를 (짝 있거나 없거나) 처리한 단계
            out.append((pa, list(used)))
        a, b = pa, pb
    return out[::-1]


def check(score_events: list[dict], shots: list[dict], max_lag: float = 20.0, lead: float = 1.0,
          start: float | None = None, end: float | None = None, replay_after: float = 30.0) -> Result:
    def rng(t):
        return (start is None or t >= start) and (end is None or t <= end)

    events = sorted((e for e in score_events if rng(float(e["t_sec"])) and e.get("flag", "ok") == "ok"), key=lambda e: float(e["t_sec"]))
    made = sorted((s for s in shots if _made(s) and rng(float(s["t_sec"]))), key=lambda s: float(s["t_sec"]))
    chosen: dict[int, list[int]] = {}
    used: set[int] = set()
    for team in sorted({e["team"] for e in events}):
        ei = [i for i, e in enumerate(events) if e["team"] == team]
        si = [i for i, s in enumerate(made) if s.get("team") == team and not _is_replay(s)]
        for a, bs in _align([events[i] for i in ei], [made[i] for i in si], max_lag, lead):
            chosen[ei[a]] = [si[b] for b in bs]
            used.update(si[b] for b in bs)
    # 리플레이에서 찍은 슛: 본 화면 슛과 짝이 안 된 변화에만, 가까운 것부터 (리플레이는 득점보다 늦게 나온다)
    for k, e in enumerate(events):
        if chosen.get(k):
            continue
        t0 = float(e["t_sec"])
        cand = [i for i, s in enumerate(made) if i not in used and _is_replay(s) and s.get("team") == e["team"]
                and t0 - max_lag <= float(s["t_sec"]) <= t0 + replay_after]
        if cand:
            i = min(cand, key=lambda i: abs(float(made[i]["t_sec"]) - t0))
            chosen[k] = [i]
            used.add(i)
    res = Result()
    res.pairs = [Pair(e, [made[i] for i in chosen.get(k, [])]) for k, e in enumerate(events)]
    res.unmatched_shots = [s for i, s in enumerate(made) if i not in used]
    return res


def read_csv(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))
