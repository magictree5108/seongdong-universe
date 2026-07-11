"""법령 심화: 조례가 인용한 국가법령을 실체화하고 위임 사슬을 잇는다.

동작:
1. 조례 644건의 조문 전문에서 「」로 인용된 국가법령명을 추출한다
   (조례·자치규칙 상호 인용은 제외 — '…법 시행규칙'만 국가령으로 인정).
2. 법제처 국가법령정보 API(law.go.kr, LAW_OC)로 각 법령을 조회해
   NationalLaw 객체를 만든다 (법령ID·구분·소관부처·공포/시행일자).
   조회 결과는 data/law_lookup.json에 캐시되어 재실행 시 API를 건너뛴다.
3. 링크 생성: NationalLaw -위임→ Ordinance
   - 제1조(목적)에서 인용 = 위임 근거로 간주, 확신도 0.95
   - 본문 다른 곳 인용 = 참조 수준, 확신도 0.7 (고정밀 경로 0.85에서 제외됨)
4. 정합성 체크: 법제처에서 확인되지 않는 인용(개정·폐지된 옛 법령명 가능성)을
   data/law_unresolved.json으로 남긴다.

객체 URL은 사용자 인증키(OC)가 들어가는 API 링크 대신 공개 항구 링크
(law.go.kr/법령/<법령명>)를 쓴다 — DB가 저장소에 커밋되기 때문.

실행: .venv/bin/python -m ontology.ingest_laws [--refresh-cache]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import tomllib
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .promote import DEFAULT_DB
from .schema import Link, LinkType, NationalLaw
from .store import OntologyStore

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "law_lookup.json"
UNRESOLVED_PATH = ROOT / "data" / "law_unresolved.json"

_LAW_RE = re.compile(r"「\s*([^」]{2,40}?)\s*」")
_PURPOSE_RE = re.compile(r"제1조\s*\(목적\)[^제]{0,300}")


def _law_oc() -> str:
    import os
    key = os.environ.get("LAW_OC")
    if not key:
        secrets = ROOT / ".streamlit" / "secrets.toml"
        if secrets.exists():
            key = tomllib.loads(secrets.read_text(encoding="utf-8")).get("LAW_OC")
    if not key:
        sys.exit("LAW_OC가 없습니다 (.streamlit/secrets.toml 또는 환경변수).")
    return key


def _is_national(name: str) -> bool:
    """자치법규(조례·자치규칙·훈령)는 제외하고 국가법령만 인정한다."""
    name = name.strip()
    # 지자체명이 들어간 것은 '…조례 시행규칙'·'…관리 규정'이라도 자치법규다
    if re.search(r"성동구|서울특별시|[가-힣]+(시|군|구)의회", name):
        return False
    if name.endswith(("조례", "규칙")) and "시행규칙" not in name:
        return False
    return name.endswith(("법", "법률", "시행령", "시행규칙", "기본법",
                          "에 관한 특별법", "규정", "영"))


def _norm(name: str) -> str:
    """공백 제거 + 가운뎃점 이형 통일 — '초·중등교육법' vs '초ㆍ중등교육법' 일치."""
    name = re.sub(r"[·ㆍ‧⋅•]", "·", name)
    return re.sub(r"\s+", "", name)


def extract_citations(store: OntologyStore):
    """조례별 (법령명, 제1조 인용 여부, 근거 발췌) 목록과 법령명별 인용 수."""
    cites: dict[str, list[tuple[str, bool, str]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for o in store.find(type="Ordinance", limit=9999):
        text = o.full_text or ""
        m = _PURPOSE_RE.search(text)
        purpose = m.group(0) if m else ""
        seen: set[str] = set()
        for cm in _LAW_RE.finditer(text):
            name = re.sub(r"\s+", " ", cm.group(1)).strip()
            if not _is_national(name) or name in seen:
                continue
            seen.add(name)
            in_purpose = name in purpose
            src_text = purpose if in_purpose else text
            i = src_text.find(name)
            evidence = re.sub(r"\s+", " ", src_text[max(0, i - 40):i + len(name) + 30])
            cites[o.id].append((name, in_purpose, evidence))
            counts[name] += 1
    return cites, counts


def resolve_law(oc: str, name: str) -> dict | None:
    """법제처 검색으로 법령 메타데이터를 얻는다. 정확 명칭 일치를 우선한다.

    짧은 이름(상법·민법 등)은 '~보상법' 같은 부분일치가 상위를 차지해
    정확 일치가 20위 밖으로 밀리므로 검색창을 더 넓힌다."""
    display = 100 if len(_norm(name)) <= 4 else 20
    params = urllib.parse.urlencode({
        "OC": oc, "target": "law", "type": "JSON", "query": name, "display": display})
    try:
        with urllib.request.urlopen(
                f"https://www.law.go.kr/DRF/lawSearch.do?{params}", timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — 일시 오류는 미해결로 남긴다
        return None
    items = data.get("LawSearch", {}).get("law", [])
    if isinstance(items, dict):
        items = [items]
    if not items:
        return None
    exact = [it for it in items if _norm(str(it.get("법령명한글", ""))) == _norm(name)]
    it = exact[0] if exact else None
    if it is None:
        # 정확 일치가 없으면 유일 결과이면서 이름을 포함할 때만 받는다 (오매칭 방지)
        if len(items) == 1 and _norm(name) in _norm(str(items[0].get("법령명한글", ""))):
            it = items[0]
        else:
            return None

    def _date(v):
        v = str(v or "")
        return f"{v[:4]}-{v[4:6]}-{v[6:8]}" if len(v) == 8 else None

    return {
        "law_id": str(it.get("법령ID", "")),
        "official_name": str(it.get("법령명한글", "")).strip(),
        "kind": str(it.get("법령구분명", "")) or None,
        "ministry": str(it.get("소관부처명", "")) or None,
        "promulgated_at": _date(it.get("공포일자")),
        "effective_at": _date(it.get("시행일자")),
        "exact": bool(exact),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--refresh-cache", action="store_true",
                    help="법제처 조회 캐시를 무시하고 전부 재조회")
    args = ap.parse_args()
    oc = _law_oc()

    cache: dict[str, dict | None] = {}
    if CACHE_PATH.exists() and not args.refresh_cache:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    with OntologyStore(args.db) as store:
        cites, counts = extract_citations(store)
        names = sorted(counts, key=lambda n: -counts[n])
        print(f"국가법령 인용: 고유 {len(names)}종, 조례 {len(cites)}건에서")

        # 법제처 조회 (캐시 우선 — 단, 이전 실패(None)는 재시도한다: 일시 오류 대비)
        fresh = 0
        for name in names:
            if cache.get(name) is not None:
                continue
            cache[name] = resolve_law(oc, name)
            fresh += 1
            if fresh % 50 == 0:
                print(f"  법제처 조회 {fresh}건…")
                CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False),
                                      encoding="utf-8")
            time.sleep(0.12)
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        resolved = {n: c for n, c in cache.items() if c}
        print(f"법제처 확인: {len(resolved)}/{len(names)}종 (신규 조회 {fresh}건)")

        # NationalLaw 객체 (같은 법령ID로 합쳐지는 이형 표기는 인용수 합산)
        laws_by_id: dict[str, NationalLaw] = {}
        name_to_lid: dict[str, str] = {}
        for name, c in resolved.items():
            lid = c["law_id"]
            name_to_lid[name] = lid
            if lid in laws_by_id:
                laws_by_id[lid].cited_count += counts[name]
                continue
            laws_by_id[lid] = NationalLaw(
                id=f"law:{lid}",
                name=c["official_name"],
                kind=c["kind"],
                ministry=c["ministry"],
                promulgated_at=c["promulgated_at"],
                effective_at=c["effective_at"],
                law_id=lid,
                cited_count=counts[name],
                source_label="법제처 국가법령정보센터",
                source_url="https://www.law.go.kr/법령/"
                           + urllib.parse.quote(c["official_name"]),
            )
        n_obj = store.upsert_objects(laws_by_id.values())

        # 위임 링크 (제1조 목적 인용 0.95 / 본문 인용 0.7)
        links, n_deleg = [], 0
        for ord_id, items in cites.items():
            for name, in_purpose, evidence in items:
                lid = name_to_lid.get(name)
                if lid is None:
                    continue
                links.append(Link(
                    type=LinkType.DELEGATES, src=f"law:{lid}", dst=ord_id,
                    method="rule", evidence=evidence[:120],
                    confidence=0.95 if in_purpose else 0.7,
                ))
                n_deleg += in_purpose
        n_link = store.upsert_links(links)

        # 정합성 체크: 법제처에서 확인 안 되는 인용 (옛 법령명·오기 가능성)
        unresolved = {n: counts[n] for n in names if not cache.get(n)}
        UNRESOLVED_PATH.write_text(
            json.dumps(unresolved, ensure_ascii=False, indent=1), encoding="utf-8")

        store.set_meta("laws_ingested_at",
                       datetime.now(KST).isoformat(timespec="seconds"))
        print(f"NationalLaw {n_obj}개 · 위임 링크 {n_link:,}개"
              f" (제1조 위임 {n_deleg}건 / 본문 참조 {n_link - n_deleg}건)")
        print(f"미확인 인용 {len(unresolved)}종 → {UNRESOLVED_PATH.name}"
              f" (개정·폐지된 옛 명칭 가능성)")
        print("객체 현황:", store.count_by_type())


if __name__ == "__main__":
    main()
