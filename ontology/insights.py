"""인사이트 엔진 — 묻기 전에 시스템이 먼저 찾아주는 발견들.

온톨로지 DB만으로(새 수집 없이) 계산 가능한 이상 신호를 발굴한다:

  1. 집행 부진   — 연도 절반이 지났는데 집행률이 바닥인 큰 사업
  2. 조기 소진   — 벌써 예산을 다 쓴 사업 (추경·이월 신호)
  3. 예산 급변   — 전년 대비 3배↑ 급증 / ⅓↓ 급감한 사업
  4. 홍보 공백   — 예산은 큰데 보도·소식 노출이 0건인 사업
  5. 조례 정비   — 개정·폐지된 옛 법령명을 인용 중인 자치법규 (별도 분석 산출물)

각 인사이트는 {category, title, detail, policy_name, url} 형태로,
앱에서 카드 + '우주에서 보기'(해당 사업 검색)로 이어진다.

예비비·보전지출·인건비성 항목은 원래 집행 패턴이 다르므로 제외한다 —
설명 가능한 규칙만 쓴다 (이상치 '탐지'가 아니라 '검토 후보' 제시).

실행(테스트): .venv/bin/python -m ontology.insights
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .promote import DEFAULT_DB
from .store import OntologyStore

ROOT = Path(__file__).resolve().parent.parent
LAW_CLEANUP_PATH = ROOT / "data" / "law_cleanup.json"

# 회계적 성격이라 집행률·홍보 잣대가 무의미한 항목
_EXCLUDE_RE = re.compile(
    r"예비비|보전지출|인력운영|기본경비|공무직|연금부담|국고보조금\s*반환|"
    r"내부거래|기금전출|시책추진|부서운영")

_won = lambda v: (f"{v/100_000_000:.1f}억" if v >= 100_000_000  # noqa: E731
                  else f"{v/10_000:,.0f}만")


def _budget_rows(store: OntologyStore) -> list[dict]:
    """BudgetItem 전체를 (연도·사업코드) 단위로 편다."""
    rows = []
    for r in store.conn.execute(
            "SELECT id, name, props FROM objects WHERE type='BudgetItem'"):
        p = json.loads(r["props"])
        parts = r["id"].split(":")          # budget:{연도}:{사업코드}:{회계}
        rows.append({
            "id": r["id"], "name": r["name"], "year": p.get("year"),
            "dbiz": parts[2] if len(parts) >= 4 else "",
            "budget": p.get("budget_current") or 0,
            "spent": p.get("expenditure") or 0,
            "field": p.get("field"),
        })
    return rows


def _mention_counts(store: OntologyStore) -> dict[str, int]:
    return {r["dst"]: r["n"] for r in store.conn.execute(
        "SELECT dst, COUNT(*) AS n FROM links WHERE type='언급' GROUP BY dst")}


def _policy_names(store: OntologyStore) -> dict[str, dict]:
    out = {}
    for r in store.conn.execute(
            "SELECT id, name, props FROM objects WHERE type='Policy'"):
        p = json.loads(r["props"])
        out[r["id"]] = {"name": r["name"], "dept": p.get("department"),
                        "budget": p.get("budget_current") or 0,
                        "field": p.get("field"), "url": p.get("source_url")}
    return out


def compute_insights(db_path: Path | str = DEFAULT_DB) -> dict:
    """카테고리별 인사이트 목록. 반환 dict는 앱이 그대로 렌더링한다."""
    with OntologyStore(db_path) as store:
        rows = _budget_rows(store)
        years = sorted({r["year"] for r in rows if r["year"]})
        cur, prev = years[-1], years[-2] if len(years) >= 2 else None
        cur_rows = [r for r in rows if r["year"] == cur
                    and not _EXCLUDE_RE.search(r["name"])]
        collected = store.get_meta("budget_exe_ymd") or ""
        asof = f"{collected[:4]}-{collected[4:6]}-{collected[6:8]}" if len(collected) == 8 else ""

        # 1) 집행 부진 — 예산 1억↑, 집행률 5%↓
        stalled = sorted(
            (r for r in cur_rows if r["budget"] >= 100_000_000
             and r["spent"] / r["budget"] < 0.05),
            key=lambda r: -r["budget"])[:6]

        # 2) 조기 소진 — 예산 5천만↑, 집행률 98%↑ (연중 기준)
        exhausted = sorted(
            (r for r in cur_rows if r["budget"] >= 50_000_000
             and r["spent"] / r["budget"] >= 0.98),
            key=lambda r: -r["budget"])[:6]

        # 3) 예산 급변 — 같은 세부사업의 전년 대비 3배↑ / ⅓↓
        swings = []
        if prev:
            by_key: dict[tuple, dict[int, int]] = defaultdict(dict)
            names: dict[tuple, str] = {}
            for r in rows:
                if _EXCLUDE_RE.search(r["name"]):
                    continue
                by_key[r["dbiz"]].setdefault(r["year"], 0)
                by_key[r["dbiz"]][r["year"]] += r["budget"]
                names[r["dbiz"]] = r["name"]
            for dbiz, per_year in by_key.items():
                b_prev, b_cur = per_year.get(prev, 0), per_year.get(cur, 0)
                if b_prev >= 100_000_000 or b_cur >= 100_000_000:
                    if b_prev and b_cur and (b_cur / b_prev >= 3 or b_cur / b_prev <= 1/3):
                        swings.append({"name": names[dbiz], "prev": b_prev,
                                       "cur": b_cur, "ratio": b_cur / b_prev})
            swings.sort(key=lambda s: -abs(s["cur"] - s["prev"]))
            swings = swings[:6]

        # 4) 홍보 공백 — 예산 5억↑ '자체사업'인데 보도·소식 언급 0건.
        #    (보조) 표기는 국비 매칭 법정급여가 대부분이라 홍보 부재가 당연 — 제외
        mentions = _mention_counts(store)
        policies = _policy_names(store)
        silent = sorted(
            ({"pid": pid, **info} for pid, info in policies.items()
             if info["budget"] >= 500_000_000 and mentions.get(pid, 0) == 0
             and "(보조)" not in info["name"]
             and not _EXCLUDE_RE.search(info["name"])),
            key=lambda x: -x["budget"])[:6]

    # 5) 조례 정비 — 사전 분석 산출물 (ontology/ingest_laws + 검증 파이프라인)
    law = {"A": [], "B": []}
    if LAW_CLEANUP_PATH.exists():
        law = json.loads(LAW_CLEANUP_PATH.read_text(encoding="utf-8"))

    return {
        "year": cur, "asof": asof,
        "stalled": [{"name": r["name"], "budget": r["budget"], "spent": r["spent"],
                     "pct": r["spent"] / r["budget"] * 100, "field": r["field"]}
                    for r in stalled],
        "exhausted": [{"name": r["name"], "budget": r["budget"], "spent": r["spent"],
                       "pct": r["spent"] / r["budget"] * 100, "field": r["field"]}
                      for r in exhausted],
        "swings": swings,
        "silent": [{"name": s["name"], "budget": s["budget"], "dept": s["dept"]}
                   for s in silent],
        "law_a": [{"old": a["old"], "new": a["new"],
                   "ordinances": [c["ord"] for c in a["cites"]]}
                  for a in law.get("A", [])],
        "law_b_count": len(law.get("B", [])),
    }


def main() -> None:
    ins = compute_insights()
    print(f"기준: {ins['year']}년 예산 · 집행은 {ins['asof']} 조회분\n")
    print(f"① 집행 부진 ({len(ins['stalled'])}건)")
    for s in ins["stalled"]:
        print(f"   {s['name'][:34]:36s} 예산 {_won(s['budget']):>8s} · 집행 {s['pct']:.1f}%")
    print(f"② 조기 소진 ({len(ins['exhausted'])}건)")
    for s in ins["exhausted"]:
        print(f"   {s['name'][:34]:36s} 예산 {_won(s['budget']):>8s} · 집행 {s['pct']:.0f}%")
    print(f"③ 예산 급변 ({len(ins['swings'])}건)")
    for s in ins["swings"]:
        print(f"   {s['name'][:34]:36s} {_won(s['prev'])} → {_won(s['cur'])} ({s['ratio']:.1f}배)")
    print(f"④ 홍보 공백 ({len(ins['silent'])}건)")
    for s in ins["silent"]:
        print(f"   {s['name'][:34]:36s} 예산 {_won(s['budget']):>8s} · {s['dept'] or '?'}")
    print(f"⑤ 조례 정비 소재: 옛 법령명 인용 {len(ins['law_a'])}종 + 표기 오류 {ins['law_b_count']}종")


if __name__ == "__main__":
    main()
