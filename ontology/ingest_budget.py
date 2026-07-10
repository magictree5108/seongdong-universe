"""지방재정365 '세부사업별 세출현황'을 수집해 Policy·BudgetItem 객체와 집행 링크를 만든다 (2단계).

출처: 지방재정365 재정데이터개방 OpenAPI (요청제한 없음, 이용조건: 출처표시·상업적 이용 가능)
  https://www.lofin365.go.kr/lf/hub/QWGJK
  파라미터: Key(인증키)·Type(json)·pIndex·pSize + fyr(회계연도)·laf_cd(자치단체코드)·exe_ymd(집행일자)
  성동구 자치단체코드: 1114000
  인증키: lofin365.go.kr 회원가입 후 '인증키 신청' (LOFIN_KEY 환경변수 또는 --key)

승격 규칙
- BudgetItem: 세부사업×회계구분 단위. id = budget:{연도}:{세부사업코드}:{회계구분코드}
  (예산현액·지출액·집행잔액은 조회일 기준 — exe_ymd를 객체에 기록)
- Policy: 세부사업 단위(연도 불문 동일 id). id = policy:{세부사업코드}
  같은 세부사업이 여러 해 반복되면 최신 연도의 예산으로 갱신. 부서는 이 API가
  부서코드(dept_cd)만 주므로 코드를 보관하고, 부서명 정합은 3단계에서 처리한다.
- 링크: Policy -집행→ BudgetItem

실행:
  LOFIN_KEY=... .venv/bin/python -m ontology.ingest_budget --fyr 2026 --exe-ymd 20260709
  .venv/bin/python -m ontology.ingest_budget --probe          # 키·응답 구조 확인용 1페이지 출력
  .venv/bin/python -m ontology.ingest_budget --sample FILE    # 저장해둔 응답 JSON으로 오프라인 실행
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .promote import DEFAULT_DB
from .schema import BudgetItem, Link, LinkType, Policy
from .store import OntologyStore

KST = timezone(timedelta(hours=9))

API_URL = "https://www.lofin365.go.kr/lf/hub/QWGJK"
SEONGDONG_LAF_CD = "1114000"
PAGE_SIZE = 500
ATTRIBUTION = "지방재정365 세부사업별 세출현황"


def _fetch_page(key: str, fyr: str, exe_ymd: str, laf_cd: str, p_index: int) -> dict:
    params = urllib.parse.urlencode({
        "Key": key, "Type": "json", "pIndex": p_index, "pSize": PAGE_SIZE,
        "fyr": fyr, "laf_cd": laf_cd, "exe_ymd": exe_ymd,
    })
    with urllib.request.urlopen(f"{API_URL}?{params}", timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_rows(payload: dict) -> tuple[list[dict], int | None]:
    """lofin365 hub 응답에서 (행 목록, 전체 건수)를 꺼낸다.

    응답 래퍼 구조가 문서화되어 있지 않아 방어적으로 파싱한다:
    - 최상위 어딘가의 {"row": [...]} 를 찾고
    - {"head": [{"list_total_count": N}, ...]} 가 있으면 전체 건수로 쓴다
    - {"RESULT": {"CODE": "ERROR-..."}} 는 예외로 올린다
    """
    def walk(node):
        if isinstance(node, dict):
            result = node.get("RESULT")
            if isinstance(result, dict) and str(result.get("CODE", "")).startswith("ERROR"):
                raise RuntimeError(f"API 오류: {result.get('CODE')} {result.get('MESSAGE')}")
            if isinstance(node.get("row"), list):
                yield ("rows", node["row"])
            if "list_total_count" in node:
                yield ("total", node["list_total_count"])
            for v in node.values():
                yield from walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v)

    rows, total = [], None
    for kind, value in walk(payload):
        if kind == "rows":
            rows.extend(value)
        elif kind == "total" and total is None:
            total = int(value)
    return rows, total


def fetch_all(key: str, fyr: str, exe_ymd: str, laf_cd: str) -> list[dict]:
    rows: list[dict] = []
    p_index = 1
    while True:
        payload = _fetch_page(key, fyr, exe_ymd, laf_cd, p_index)
        page, total = _extract_rows(payload)
        rows.extend(page)
        if not page or (total is not None and len(rows) >= total) or len(page) < PAGE_SIZE:
            return rows
        p_index += 1


def _amount(row: dict, field: str) -> int | None:
    v = row.get(field)
    if v in (None, ""):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def build_objects(rows: list[dict]) -> tuple[list[BudgetItem], list[Policy], list[Link]]:
    """행 → BudgetItem·Policy·집행 링크. 같은 세부사업의 여러 회계구분은 합산해 Policy에 반영."""
    budget_items: dict[str, BudgetItem] = {}
    policy_rows: dict[str, list[dict]] = {}

    for row in rows:
        dbiz_cd = str(row.get("dbiz_cd", "")).strip()
        fyr = str(row.get("fyr", "")).strip()
        if not dbiz_cd or not fyr:
            continue
        acnt_cd = str(row.get("acnt_dv_cd", "")).strip() or "NA"
        bdg = _amount(row, "bdg_cash_amt")
        ep = _amount(row, "ep_amt")
        item_id = f"budget:{fyr}:{dbiz_cd}:{acnt_cd}"
        budget_items[item_id] = BudgetItem(
            id=item_id,
            name=str(row.get("dbiz_nm", "")).strip() or dbiz_cd,
            account=row.get("acnt_dv_nm"),
            field=row.get("fld_nm"),
            budget_current=bdg,
            expenditure=ep,
            balance=(bdg - ep) if (bdg is not None and ep is not None) else None,
            year=int(fyr) if fyr.isdigit() else None,
            source_label=f"{ATTRIBUTION} (집행일자 {row.get('exe_ymd', '?')})",
            source_url="https://www.lofin365.go.kr/portal/LF5110000.do",
        )
        policy_rows.setdefault(dbiz_cd, []).append(row)

    policies: list[Policy] = []
    links: list[Link] = []
    for dbiz_cd, rws in policy_rows.items():
        latest_fyr = max(str(r.get("fyr", "")) for r in rws)
        latest = [r for r in rws if str(r.get("fyr", "")) == latest_fyr]
        head = latest[0]
        bdg = sum(_amount(r, "bdg_cash_amt") or 0 for r in latest)
        ep = sum(_amount(r, "ep_amt") or 0 for r in latest)
        pid = f"policy:{dbiz_cd}"
        policies.append(Policy(
            id=pid,
            name=str(head.get("dbiz_nm", "")).strip() or dbiz_cd,
            department=None,  # 이 API는 부서코드만 제공 — 부서명 정합은 3단계
            dept_code=str(head.get("dept_cd", "")).strip() or None,
            field=head.get("fld_nm"),
            budget_current=bdg or None,
            expenditure=ep or None,
            year=int(latest_fyr) if latest_fyr.isdigit() else None,
            source_label=ATTRIBUTION,
            source_url="https://www.lofin365.go.kr/portal/LF5110000.do",
        ))
        for r in rws:
            fyr = str(r.get("fyr", "")).strip()
            acnt_cd = str(r.get("acnt_dv_cd", "")).strip() or "NA"
            links.append(Link(
                type=LinkType.EXECUTES, src=pid,
                dst=f"budget:{fyr}:{dbiz_cd}:{acnt_cd}",
                method="rule", evidence="세부사업코드 일치",
            ))
    return list(budget_items.values()), policies, links


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--key", default=os.environ.get("LOFIN_KEY"))
    ap.add_argument("--fyr", default=str(datetime.now(KST).year))
    ap.add_argument("--exe-ymd",
                    default=(datetime.now(KST) - timedelta(days=1)).strftime("%Y%m%d"))
    ap.add_argument("--laf-cd", default=SEONGDONG_LAF_CD)
    ap.add_argument("--out", type=Path, default=DEFAULT_DB)
    ap.add_argument("--probe", action="store_true",
                    help="1페이지 원시 응답만 출력 (키·구조 확인)")
    ap.add_argument("--sample", type=Path,
                    help="저장된 응답 JSON 파일로 오프라인 실행 (API 미호출)")
    args = ap.parse_args()

    if args.sample:
        payload = json.loads(args.sample.read_text(encoding="utf-8"))
        rows, _ = _extract_rows(payload)
    else:
        if not args.key:
            sys.exit("인증키가 없습니다 — LOFIN_KEY 환경변수 또는 --key 로 전달하십시오.\n"
                     "발급: lofin365.go.kr 로그인 → 데이터셋 페이지 '인증키 신청'")
        if args.probe:
            payload = _fetch_page(args.key, args.fyr, args.exe_ymd, args.laf_cd, 1)
            print(json.dumps(payload, ensure_ascii=False, indent=1)[:3000])
            return
        rows = fetch_all(args.key, args.fyr, args.exe_ymd, args.laf_cd)

    print(f"수집: {len(rows)}행 (회계연도 {args.fyr}, 집행일자 {args.exe_ymd})")
    items, policies, links = build_objects(rows)

    with OntologyStore(args.out) as store:
        n_obj = store.upsert_objects(policies + items)
        n_link = store.upsert_links(links)
        store.set_meta("budget_ingested_at", datetime.now(KST).isoformat(timespec="seconds"))
        store.set_meta("budget_exe_ymd", args.exe_ymd)
        counts = store.count_by_type()
        link_counts = store.count_links_by_type()

    print(f"승격: Policy {len(policies):,} · BudgetItem {len(items):,} · 집행링크 {n_link:,}")
    print(f"저장소 현황: {counts}")
    print(f"링크 현황: {link_counts}")


if __name__ == "__main__":
    main()
