"""3단계: 그래프 구축 — 부서 정합 + 조례↔사업 + 보도↔사업 링크 생성.

외부에 부서코드→부서명 공개 사전이 없으므로(지방재정365는 dept_cd만 제공),
부서코드별 사업명 목록을 증거로 Claude가 조직·업무분장 41개 부서에 정합한다.
조례·보도 링크는 어휘 후보 생성(토큰 겹침) 후 Claude가 확정한다 — 후보가 없는
쌍은 애초에 판정하지 않아 호출 수와 오탐을 함께 줄인다.

모든 링크는 method='claude'/'rule', evidence(판정 근거), confidence를 지니며
--stage report 가 수기 검증용 표본 리포트를 만든다 (설계서 3단계 벤치마크).

실행 (순서대로):
  .venv/bin/python -m ontology.build_links --stage dept       # 부서 사전 + 담당 링크
  .venv/bin/python -m ontology.build_links --stage ordinance  # 근거 링크
  .venv/bin/python -m ontology.build_links --stage press      # 언급 링크
  .venv/bin/python -m ontology.build_links --stage report     # 수기 검증 리포트

ANTHROPIC_API_KEY는 .streamlit/secrets.toml 또는 환경변수에서 읽는다.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tomllib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .promote import DEFAULT_DB
from .schema import Link, LinkType
from .store import OntologyStore

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parent.parent
DEPT_MAP_PATH = ROOT / "data" / "dept_map.json"
REPORT_PATH = ROOT / "data" / "link_review.md"

MODEL = "claude-sonnet-5"
MAX_TOKENS = 4000          # Claude 5는 thinking 블록이 토큰을 먼저 소모 — 여유 필수
WORKERS = 6
MIN_CONFIDENCE = 0.7

_STOPWORDS = "서울특별시|성동구|조례|규칙|시행|에 관한|지원|운영|관리|사업|추진|등"


def _tokens(s: str) -> set[str]:
    s = re.sub(_STOPWORDS, " ", s)
    return {t for t in re.split(r"[^가-힣A-Za-z0-9]+", s) if len(t) >= 2}


# ── Claude 클라이언트 ────────────────────────────────────────────


def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        secrets = ROOT / ".streamlit" / "secrets.toml"
        if secrets.exists():
            key = tomllib.loads(secrets.read_text(encoding="utf-8")).get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY가 없습니다 (.streamlit/secrets.toml 또는 환경변수).")
    return key


def _make_client():
    import anthropic
    return anthropic.Anthropic(api_key=_api_key())


def _ask_json(client, prompt: str):
    """Claude에 묻고 응답에서 JSON을 꺼낸다. 텍스트 블록만 취하고 여분 산문은 무시."""
    msg = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"JSON 없음: {text[:200]}")
    return json.loads(m.group(0))


def _run_batches(batches: list[str], label: str) -> list:
    """프롬프트 배치를 동시 실행하고 JSON 결과 목록을 돌려준다 (실패 배치는 건너뜀)."""
    client = _make_client()
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_ask_json, client, b): i for i, b in enumerate(batches)}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"  ⚠ 배치 {futures[fut]} 실패: {e}", file=sys.stderr)
            if done % 20 == 0 or done == len(batches):
                print(f"  {label}: {done}/{len(batches)} 배치")
    return results


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _resolve(id_of: dict, raw_id, prefix: str):
    """모델이 접두어를 벗겨 반환하는 경우가 있어 관대하게 해석한다."""
    rid = str(raw_id or "").strip().strip("[]")
    return id_of.get(rid) or id_of.get(f"{prefix}:{rid}")


# ── Stage A: 부서 정합 ───────────────────────────────────────────


def stage_dept(store: OntologyStore) -> None:
    pols = store.find(type="Policy", limit=99999)
    depts = store.find(type="Department", limit=100)
    by_code: dict[str, list] = defaultdict(list)
    for p in pols:
        if p.dept_code:
            by_code[p.dept_code].append(p)

    dept_list = "\n".join(
        f"- {d.name}: {(d.duties or '')[:150]}" for d in sorted(depts, key=lambda x: x.name)
    )
    codes = sorted(by_code, key=lambda c: -len(by_code[c]))
    batches = []
    for group in _chunks(codes, 8):
        lines = []
        for code in group:
            names = sorted({p.name for p in by_code[code]})[:20]
            lines.append(f"부서코드 {code} ({len(by_code[code])}개 사업): {', '.join(names)}")
        batches.append(f"""성동구 예산 시스템의 부서코드를 실제 부서에 매칭하라.

[성동구 부서 목록 (41개, 이름: 핵심업무)]
{dept_list}

[매칭할 부서코드와 그 부서 소관 사업명들]
{chr(10).join(lines)}

각 부서코드의 사업명들이 어느 부서의 업무분장에 해당하는지 판단하라.
동주민센터·보건소·의회 등 목록에 없는 조직의 사업이면 department를 null로 하라.
JSON만 출력: {{"<부서코드>": {{"department": "<부서명 또는 null>", "confidence": 0.0~1.0, "reason": "<한 문장>"}}, ...}}""")

    results = _run_batches(batches, "부서 정합")
    dept_map: dict[str, dict] = {}
    for r in results:
        dept_map.update(r)

    by_name = {d.name: d for d in depts}
    links, updated = [], 0
    for code, m in dept_map.items():
        dept = by_name.get(m.get("department") or "")
        if dept is None or m.get("confidence", 0) < MIN_CONFIDENCE:
            continue
        for p in by_code.get(code, []):
            links.append(Link(
                type=LinkType.MANAGES, src=dept.id, dst=p.id, method="claude",
                evidence=m.get("reason"), confidence=float(m.get("confidence", 0)),
            ))
            p.department = dept.name
            store.upsert_object(p, commit=False)
            updated += 1
    store.conn.commit()
    n = store.upsert_links(links)

    DEPT_MAP_PATH.write_text(
        json.dumps(dept_map, ensure_ascii=False, indent=1), encoding="utf-8")
    matched = sum(1 for m in dept_map.values()
                  if m.get("department") and m.get("confidence", 0) >= MIN_CONFIDENCE)
    print(f"부서 사전: {len(dept_map)}코드 중 {matched}개 정합 → {DEPT_MAP_PATH.name}")
    print(f"담당 링크 {n:,}개 생성, Policy.department 채움 {updated:,}건")


# ── Stage B: 조례 -근거→ 사업 ────────────────────────────────────


def stage_ordinance(store: OntologyStore) -> None:
    pols = store.find(type="Policy", limit=99999)
    ords = store.find(type="Ordinance", limit=9999)
    pol_tokens = [(p, _tokens(p.name)) for p in pols]

    tasks = []
    for o in ords:
        ot = _tokens(o.name)
        if not ot:
            continue
        scored = sorted(
            ((len(ot & pt) / len(ot | pt), p) for p, pt in pol_tokens if ot & pt),
            key=lambda x: -x[0],
        )[:8]
        if scored:
            tasks.append((o, [p for _, p in scored]))
    print(f"조례 {len(ords)}개 중 후보 보유 {len(tasks)}개")

    id_of = {}
    batches = []
    for group in _chunks(tasks, 6):
        lines = []
        for o, cands in group:
            purpose = re.search(r"제1조[^제]{0,200}", o.full_text or "")
            lines.append(
                f"조례 [{o.id}] {o.name}\n  목적: {(purpose.group(0) if purpose else '')[:180]}\n"
                f"  후보 사업: " + " / ".join(f"[{p.id}] {p.name}" for p in cands))
            id_of[o.id] = o
            for p in cands:
                id_of[p.id] = p
        batches.append(f"""성동구 조례가 예산 사업의 '법적 근거'인지 판정하라.

근거로 인정하는 경우:
- 조례가 그 사업(시설·센터·위원회·구단 등)의 설치·운영을 규정 → 그 운영/지원 사업
- 조례가 지원·수당·기금·재정지원 제도를 규정 → 그 제도를 집행하는 사업
  (예: '마을버스 재정지원 조례'는 마을버스 지원·운영 사업들의 근거다)
근거가 아닌 경우: 낱말만 겹치고 규율 대상이 다른 사업 (예: '교통안전 조례'와 무관한 도로포장 공사)

확신도 눈금: 0.9~1.0 조례가 사업 자체를 명시 / 0.7~0.85 조례가 규정한 제도의 집행 사업 /
0.5~0.65 개연성은 있으나 불확실(보고는 하되 낮게) / 그 미만은 생략.

{chr(10).join(lines)}

JSON만 출력: [{{"ordinance": "<조례 id>", "policy": "<사업 id>", "confidence": 0.0~1.0, "reason": "<한 문장>"}}, ...] (없으면 [])""")

    results = _run_batches(batches, "근거 판정")
    raw = [m for r in results if isinstance(r, list) for m in r]
    (ROOT / "data" / "link_judgments_ordinance.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    links = []
    for r in results:
        for m in r if isinstance(r, list) else []:
            o = _resolve(id_of, m.get("ordinance"), "ordinance")
            p = _resolve(id_of, m.get("policy"), "policy")
            if (o is None or p is None or type(o).__name__ != "Ordinance"
                    or type(p).__name__ != "Policy"
                    or m.get("confidence", 0) < MIN_CONFIDENCE):
                continue
            links.append(Link(
                type=LinkType.BASIS_OF, src=o.id, dst=p.id, method="claude",
                evidence=m.get("reason"), confidence=float(m.get("confidence", 0)),
            ))
            if p.basis_ordinance is None:
                p.basis_ordinance = o.name
                store.upsert_object(p, commit=False)
    store.conn.commit()
    n = store.upsert_links(links)
    print(f"근거 링크 {n:,}개 생성 (조례 {len({l.src for l in links})}개 ↔ 사업 {len({l.dst for l in links})}개)")


# ── Stage C: 보도 -언급→ 사업 ────────────────────────────────────


def stage_press(store: OntologyStore) -> None:
    pols = store.find(type="Policy", limit=99999)
    press = store.find(type="PressRelease", limit=99999)
    pol_tokens = [(p, _tokens(p.name)) for p in pols if _tokens(p.name)]

    tasks = []
    for pr in press:
        body = (pr.name + " " + (pr.body or ""))[:3000]
        cands = []
        for p, pt in pol_tokens:
            hit = sum(1 for t in pt if t in body)
            if hit / len(pt) >= 0.6:
                cands.append((hit / len(pt), p))
        cands.sort(key=lambda x: -x[0])
        if cands:
            tasks.append((pr, [p for _, p in cands[:6]]))
    print(f"보도/소식 {len(press)}개 중 후보 보유 {len(tasks)}개")

    id_of = {}
    batches = []
    for group in _chunks(tasks, 5):
        lines = []
        for pr, cands in group:
            id_of[pr.id] = pr
            for p in cands:
                id_of[p.id] = p
            lines.append(
                f"보도 [{pr.id}] {pr.name} ({pr.published_at})\n"
                f"  본문: {(pr.body or '')[:700]}\n"
                f"  후보 사업: " + " / ".join(f"[{p.id}] {p.name}" for p in cands))
        batches.append(f"""성동구 보도자료가 예산 사업을 실제로 다루는지 판정하라.
'언급'이란 보도 내용이 해당 사업의 시행·모집·성과를 알리는 경우다.
낱말만 겹치고 내용이 다른 경우는 제외하라. 확실한 쌍만 보고하라.

{chr(10).join(lines)}

JSON만 출력: [{{"press": "<보도 id>", "policy": "<사업 id>", "confidence": 0.0~1.0, "reason": "<한 문장>"}}, ...] (없으면 [])""")

    results = _run_batches(batches, "언급 판정")
    raw = [m for r in results if isinstance(r, list) for m in r]
    (ROOT / "data" / "link_judgments_press.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    links = []
    mentioned: dict[str, list[str]] = defaultdict(list)
    for r in results:
        for m in r if isinstance(r, list) else []:
            pr = _resolve(id_of, m.get("press"), "press")
            p = _resolve(id_of, m.get("policy"), "policy")
            if (pr is None or p is None or type(pr).__name__ != "PressRelease"
                    or type(p).__name__ != "Policy"
                    or m.get("confidence", 0) < MIN_CONFIDENCE):
                continue
            links.append(Link(
                type=LinkType.MENTIONS, src=pr.id, dst=p.id, method="claude",
                evidence=m.get("reason"), confidence=float(m.get("confidence", 0)),
            ))
            mentioned[pr.id].append(p.name)
    for pid, names in mentioned.items():
        pr = store.get(pid)
        pr.mentioned_policies = sorted(set(names))
        store.upsert_object(pr, commit=False)
    store.conn.commit()
    n = store.upsert_links(links)
    print(f"언급 링크 {n:,}개 생성 (보도 {len(mentioned):,}개)")


# ── Stage D: 수기 검증 리포트 ────────────────────────────────────


def stage_report(store: OntologyStore) -> None:
    random.seed(50)
    rows = store.conn.execute(
        "SELECT type, src, dst, props FROM links WHERE type IN ('담당','근거','언급')"
    ).fetchall()
    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    lines = [
        "# 온톨로지 링크 수기 검증 표본 (3단계 벤치마크)",
        f"\n생성: {datetime.now(KST).isoformat(timespec='seconds')}",
        "\n각 링크의 근거(evidence)를 보고 O/X를 표시해 정확도를 계산하십시오.\n",
    ]
    for t, pool in sorted(by_type.items()):
        picks = random.sample(pool, min(17, len(pool)))
        lines.append(f"\n## {t} 링크 (전체 {len(pool):,}개 중 {len(picks)}개 표본)\n")
        lines.append("| # | 출발 | 도착 | 확신도 | 근거 | 판정(O/X) |")
        lines.append("|---|---|---|---|---|---|")
        for i, r in enumerate(picks, 1):
            src, dst = store.get(r["src"]), store.get(r["dst"])
            props = json.loads(r["props"])
            ev = (props.get("evidence") or "").replace("|", "/")[:80]
            lines.append(f"| {i} | {src.name[:30]} | {dst.name[:30]} |"
                         f" {props.get('confidence', '?')} | {ev} | |")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"검증 리포트 → {REPORT_PATH} ({sum(min(17, len(v)) for v in by_type.values())}건)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["dept", "ordinance", "press", "report"])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args()
    with OntologyStore(args.db) as store:
        {"dept": stage_dept, "ordinance": stage_ordinance,
         "press": stage_press, "report": stage_report}[args.stage](store)
        store.set_meta(f"links_{args.stage}_at",
                       datetime.now(KST).isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
