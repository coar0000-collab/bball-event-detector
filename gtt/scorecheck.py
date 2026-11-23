"""코트만으로 계산한 득점이 스코어버그(정답)와 얼마나 맞는지 잰다.

목표 지표: 스코어버그를 보지 않고 슛 위치(코트 라인·거리)로 정한 자유투/2점/3점이
스코어버그의 +1/+2/+3 과 일치하는 비율. 스코어버그 점수 변화는 정답으로만 쓴다.

짝짓기: 스코어버그 점수 변화 하나마다, 같은 팀의 '성공' 슛 중 그 변화보다 max_lag 초 안쪽으로 앞서는
(또는 lead 초 안쪽으로 뒤따르는) 가장 가까운 슛을 고른다. 스코어버그가 자유투 2개를 한 번에 +2 로 올리면
자유투 2개와 짝짓는다.
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
        agree = [p for p in matched if p.agree]
        conf: dict[str, int] = {}
        for p in matched:
            k = f"코트 {p.court_points} / 점수판 {p.bug_points}"
            conf[k] = conf.get(k, 0) + 1
        near = [p for p in matched if any(str(s.get("near_line")).lower() == "true" for s in p.shots)]
        totals = {}
        for team in ("home", "away"):
            bug = sum(p.bug_points for p in self.pairs if p.event["team"] == team)
            court = sum(shot_points(s) or 0 for p in matched if p.event["team"] == team for s in p.shots)
            court += sum(shot_points(s) or 0 for s in self.unmatched_shots if s.get("team") == team)
            totals[team] = {"점수판": bug, "코트": court}
        return {
            "점수판 변화": len(self.pairs),
            "짝지은 변화": len(matched),
            "일치": len(agree),
            "일치율(짝지은 것 중)": round(len(agree) / len(matched), 3) if matched else None,
            "일치율(전체 변화 중)": round(len(agree) / len(self.pairs), 3) if self.pairs else None,
            "슛 없이 바뀐 점수": len(self.pairs) - len(matched),
            "점수 변화 없는 성공 슛": len(self.unmatched_shots),
            "라인 근처 슛이 낀 변화": len(near),
            "라인 근처에서 틀림": sum(1 for p in near if not p.agree),
            # 라인에서 충분히 떨어져 코트만으로 확실히 정한 것. 여기서 틀리면 추적·찍기·짝짓기 문제다.
            "확실한 판정": len(matched) - len(near),
            "확실한 판정 일치": sum(1 for p in matched if p not in near and p.agree),
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


def check(score_events: list[dict], shots: list[dict], max_lag: float = 8.0, lead: float = 1.0,
          start: float | None = None, end: float | None = None) -> Result:
    def rng(t):
        return (start is None or t >= start) and (end is None or t <= end)

    events = sorted((e for e in score_events if rng(float(e["t_sec"])) and e.get("flag", "ok") == "ok"), key=lambda e: float(e["t_sec"]))
    made = sorted((s for s in shots if _made(s) and rng(float(s["t_sec"]))), key=lambda s: float(s["t_sec"]))
    used: set[int] = set()
    res = Result()
    for e in events:
        te = float(e["t_sec"])
        cand = [i for i, s in enumerate(made) if i not in used and s.get("team") == e["team"]
                and te - max_lag <= float(s["t_sec"]) <= te + lead]
        cand.sort(key=lambda i: abs(te - float(made[i]["t_sec"])))
        chosen: list[int] = []
        if int(e["delta"]) == 2:  # 자유투 2개가 한 번에 오른 경우
            fts = [i for i in cand if shot_points(made[i]) == 1]
            if len(fts) >= 2 and not any(shot_points(made[i]) == 2 for i in cand[:1]):
                chosen = sorted(fts[:2])
        if not chosen and cand:
            chosen = [cand[0]]
        used.update(chosen)
        res.pairs.append(Pair(e, [made[i] for i in chosen]))
    res.unmatched_shots = [s for i, s in enumerate(made) if i not in used]
    return res


def read_csv(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))
