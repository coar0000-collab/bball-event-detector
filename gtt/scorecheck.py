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


def resolve_corrections(rows: list[dict], window: float = 30.0, lookback: float = 300.0) -> list[dict]:
    """기록원이 점수를 고친 흔적(감소)을 앞뒤 변화와 묶어 정리한다. 원본은 두고 새 목록을 돌려준다.

    - 감소 뒤 window 초 안에 같은 팀이 같은 만큼 다시 오르면(잠깐 잘못 내렸다가 되돌림) 그 증가는 득점이 아니다
      → flag "revert". (실제 영상 73.5s·87.5s 에서 '+1 ok' 로 나와 자유투 하나가 있는 것처럼 보였다)
    - 아니면, 감소 바로 앞의 같은 팀 정상 변화(window 초 안)를 고친 것으로 본다. 합이 0 이면 그 변화는 취소
      (flag "reverted"), 1~3 이면 고친 값으로 바꾼다(+3 → -1 이면 +2). 감소 자체는 flag "merged".
    - 그것도 아니면 늦게 몰아서 고친 것으로 본다: 감소 앞 lookback 초 안의 같은 팀 정상 변화를 최근 것부터 더해 가다
      감소량과 딱 맞으면 그 변화들을 모두 취소(flag "reverted", note "늦은 정정"). 실제 영상: 147s·173s 에 슛 없이 +1 이
      두 번 올랐다가 341s 에 -2 로 한꺼번에 지워졌다.
    정상(ok)만 득점으로 세므로 score-check 는 고친 뒤의 점수로 비교한다.

    먼저 '오독 구간'을 합친다: 같은 팀에서 말이 안 되는 변화(check/correction, 3점보다 큼)가 window 초 안에 이어지고
    그 합이 1~3 이면 점수판이 한 번 바뀐 것을 잘못 읽은 것이다 → 첫 변화를 그 합의 정상 득점으로, 나머지는 "merged".
    실제 영상: 67→69 를 67→'6'(-61)→69(+63) 로 읽음, 22→24 를 '114'(+92)→24(-90) 로 읽음. 합이 0 이면 모두 "merged"."""
    out = [dict(r) for r in rows]
    order = sorted(range(len(out)), key=lambda i: float(out[i]["t_sec"]))
    for team in {r["team"] for r in out}:
        idx = [i for i in order if out[i]["team"] == team]
        p = 0
        while p < len(idx):
            e = out[idx[p]]
            if e.get("flag") not in ("check", "correction") or abs(int(e["delta"])) <= 3:
                p += 1
                continue
            t0, acc, run = float(e["t_sec"]), int(e["delta"]), [idx[p]]
            q = p + 1
            while q < len(idx) and len(run) < 4:
                r = out[idx[q]]
                if r.get("flag") not in ("check", "correction") or float(r["t_sec"]) - t0 > window:
                    break
                acc += int(r["delta"])
                run.append(idx[q])
                q += 1
                if 0 <= acc <= 3:
                    break
            if len(run) >= 2 and 0 <= acc <= 3:
                deltas = ", ".join(f"{int(out[i]['delta']):+d}" for i in run)
                for i in run[1:]:
                    out[i]["flag"] = "merged"
                if acc == 0:
                    e["flag"] = "merged"
                else:
                    e["flag"], e["delta"], e["note"] = "ok", acc, f"오독 구간 합침({deltas})"
                p = q
            else:
                p += 1
        for p, j in enumerate(idx):
            e = out[j]
            d = int(e["delta"])
            if d >= 0 or e.get("flag") != "correction":
                continue
            t = float(e["t_sec"])
            nxt = idx[p + 1] if p + 1 < len(idx) else None
            if nxt is not None and int(out[nxt]["delta"]) == -d and float(out[nxt]["t_sec"]) - t <= window:
                out[nxt]["flag"] = "revert"
                continue
            prv = idx[p - 1] if p > 0 else None
            if prv is not None and out[prv].get("flag", "ok") == "ok" and t - float(out[prv]["t_sec"]) <= window:
                net = int(out[prv]["delta"]) + d
                if net == 0:
                    out[prv]["flag"] = "reverted"
                    e["flag"] = "merged"
                    continue
                if net in (1, 2, 3):
                    out[prv]["note"] = f"{out[prv]['delta']}→{net} 정정"
                    out[prv]["delta"] = net
                    e["flag"] = "merged"
                    continue
            # 늦게 몰아서 고침: 최근 정상 변화부터 더해 감소량과 딱 맞는 묶음
            acc, chosen = 0, []
            for q in reversed(idx[:p]):
                r = out[q]
                if t - float(r["t_sec"]) > lookback:
                    break
                if r.get("flag", "ok") != "ok" or int(r["delta"]) <= 0:
                    continue
                acc += int(r["delta"])
                chosen.append(q)
                if acc >= -d:
                    break
            if acc == -d and chosen:
                for q in chosen:
                    out[q]["flag"] = "reverted"
                    out[q]["note"] = f"늦은 정정({e.get('time', round(t, 1))} {d:+d})으로 취소"
                e["flag"] = "merged"
    return out


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
            "라인 근처 사람 판정": sum(1 for p in near if any(s.get("line_call") for s in p.shots)),
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
                "라인 판정": s0.get("line_call", ""), "자유투 자리": s0.get("ft_spot", ""),
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

    # 기록원 정정(감소)을 앞뒤 변화와 묶은 뒤 정상 득점만 (이미 정리된 CSV 에 다시 해도 결과가 같다)
    fixed = resolve_corrections([e for e in score_events if "flag" in e]) + [e for e in score_events if "flag" not in e]
    events = sorted((e for e in fixed if rng(float(e["t_sec"])) and e.get("flag", "ok") == "ok"), key=lambda e: float(e["t_sec"]))
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
