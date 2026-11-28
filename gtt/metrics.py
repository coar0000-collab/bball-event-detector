"""선수 평가지표를 설정 파일(JSON)의 수식으로 계산한다.

수식은 코드가 아니라 설정 파일에 있어서, 확정된 수식이 오면 JSON 만 바꾸면 된다.
수식에 쓸 수 있는 기록 이름은 STATS 를 보면 된다. 사칙연산·괄호·숫자·params 의 이름, min/max 만 허용한다
(eval 을 쓰지 않고 직접 해석하므로 임의 코드는 실행되지 않는다).

지금 영상으로 재지 못한 기록(리바운드·어시스트 등)이 수식에 들어 있으면 그 지표는 '계산 불가'로 두고,
참고용으로 없는 기록을 0으로 둔 '부분값'과 빠진 기록 목록을 함께 낸다.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

# 수식에서 쓰는 이름 → 한글 설명
STATS = {
    "PTS": "득점", "FGM": "야투 성공", "FGA": "야투 시도", "P2M": "2점 성공", "P2A": "2점 시도",
    "P3M": "3점 성공", "P3A": "3점 시도", "FTM": "자유투 성공", "FTA": "자유투 시도",
    "ORB": "공격 리바운드", "DRB": "수비 리바운드", "REB": "리바운드", "AST": "어시스트",
    "STL": "스틸", "BLK": "블락", "TOV": "턴오버", "PF": "파울",
}

_FUNCS = {"min": min, "max": max}


@dataclass
class Metric:
    key: str
    name: str
    formula: str
    params: dict
    note: str = ""
    fmt: str = "{:.1f}"

    def names(self) -> set[str]:
        tree = ast.parse(self.formula, mode="eval")
        return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - set(self.params) - set(_FUNCS)

    def needs(self) -> list[str]:
        return sorted(self.names() & set(STATS), key=list(STATS).index)


def _eval(node, env):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise KeyError(node.id)
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a, b = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if b == 0:
            raise ZeroDivisionError
        return a / b
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(x, env) for x in node.args])
    raise ValueError(f"수식에 쓸 수 없는 표현: {ast.dump(node)[:60]}")


def validate(m: Metric) -> None:
    tree = ast.parse(m.formula, mode="eval")
    unknown = m.names() - set(STATS)
    if unknown:
        raise ValueError(f"지표 '{m.name}' 수식에 모르는 이름 {sorted(unknown)} (쓸 수 있는 기록: {', '.join(STATS)}; 가중치는 params 에)")
    env = {k: 1.0 for k in STATS} | {k: float(v) for k, v in m.params.items()}
    try:
        _eval(tree, env)
    except ZeroDivisionError:
        pass


def load(path: str | Path | None = None) -> tuple[list[Metric], dict]:
    """설정 파일을 읽는다. path 가 없으면 패키지에 든 기본 설정(gtt/metrics_default.json)."""
    if path:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        d = json.loads(resources.files("gtt").joinpath("metrics_default.json").read_text(encoding="utf-8"))
    out = []
    for m in d["metrics"]:
        mm = Metric(key=m["key"], name=m.get("name", m["key"]), formula=m["formula"], params=m.get("params", {}),
                    note=m.get("note", ""), fmt=m.get("format", "{:.1f}"))
        validate(mm)
        out.append(mm)
    return out, d


def evaluate(m: Metric, stats: dict, available: set[str]) -> dict:
    """한 선수(또는 팀)의 지표. stats: 기록 이름 → 값, available: 영상으로 잰 기록 이름."""
    needs = m.needs()
    missing = [k for k in needs if k not in available]
    env = {k: float(stats.get(k) or 0) for k in STATS} | {k: float(v) for k, v in m.params.items()}
    try:
        v = _eval(ast.parse(m.formula, mode="eval"), env)
    except ZeroDivisionError:
        v = None
    if v is not None:
        v = round(v, 4)
    text = "-" if v is None else m.fmt.format(v)
    if missing:
        return {"value": None, "partial": v, "text": "계산 불가", "partial_text": text,
                "missing": missing, "missing_kr": [STATS[k] for k in missing]}
    return {"value": v, "partial": v, "text": text, "partial_text": text, "missing": [], "missing_kr": []}
