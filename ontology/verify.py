"""1단계 벤치마크 — 가이드라인 기준: "스키마로부터 빈 그래프 생성·조회 성공, 더미 객체 1개 왕복."

세 가지를 검증한다:
1. 빈 저장소 왕복  — 스키마로 빈 DB 생성, 더미 Policy 1개 저장→조회→동일성 확인→삭제.
                     재승격(재upsert) 시 링크가 보존되는지 회귀 검증 포함.
2. 링크 규칙 검증  — 8개 링크 타입의 endpoint 제약이 실제로 잘못된 링크를 거부하는지
3. 실데이터 다중 홉 — 승격 DB의 '사본'에서 실제 조례·부서에 더미 Policy를 걸고
                     "이 사업의 근거조례와 담당부서는?" 을 링크로 답한다.
                     운영 DB(data/ontology.db)에는 어떤 쓰기도 하지 않는다.

실행: .venv/bin/python -m ontology.verify [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from .promote import DEFAULT_DB
from .schema import Department, Link, LinkType, Ordinance, Policy
from .store import OntologyStore


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "✅" if ok else "❌"
    print(f"{mark} {label}" + (f" — {detail}" if detail else ""))
    return ok


def verify_empty_roundtrip() -> bool:
    """벤치마크 1·2: 빈 그래프 생성·더미 객체 왕복·링크 규칙."""
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        with OntologyStore(Path(tmp) / "empty.db") as store:
            ok &= check("빈 그래프 생성·조회", store.count_by_type() == {})

            dummy = Policy(
                id="policy:dummy-roundtrip",
                name="더미 사업 (왕복 테스트)",
                department="테스트과",
                field="테스트",
                budget_current=1_000_000,
                year=2026,
            )
            store.upsert_object(dummy)
            back = store.get(dummy.id)
            ok &= check("더미 Policy 왕복 (저장→조회→동일)", back == dummy,
                        f"{type(back).__name__} '{back.name if back else '?'}'")

            dept = Department(id="department:dummy", name="더미과")
            store.upsert_object(dept)
            store.upsert_link(Link(type=LinkType.MANAGES, src=dept.id, dst=dummy.id))
            ok &= check("올바른 링크 허용 (Department -담당→ Policy)",
                        len(store.links_of(dummy.id)) == 1)

            try:
                store.upsert_link(Link(type=LinkType.MANAGES, src=dummy.id, dst=dept.id))
                ok &= check("잘못된 링크 거부 (Policy -담당→ Department)", False)
            except ValueError as e:
                ok &= check("잘못된 링크 거부 (Policy -담당→ Department)", True, str(e))

            # 회귀: 재승격(같은 id 재upsert)이 기존 링크를 지우면 안 된다
            # (INSERT OR REPLACE + ON DELETE CASCADE 결합 버그 방지)
            store.upsert_object(dummy)
            ok &= check("재upsert 시 링크 보존 (재승격 안전성)",
                        len(store.links_of(dummy.id)) == 1)

            store.delete(dummy.id)
            ok &= check("삭제 시 링크 연쇄 정리 (ON DELETE CASCADE)",
                        store.get(dummy.id) is None and store.links_of(dept.id) == [])
    return ok


def verify_real_multihop(db_path: Path) -> bool:
    """벤치마크 3: 승격된 실데이터의 '사본' 위에서 더미 Policy 다중 홉.

    운영 DB에는 쓰기를 하지 않는다 — 더미가 남거나 검증이 중단돼도 무해하다.
    """
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        copy_path = Path(tmp) / "ontology-copy.db"
        shutil.copy(db_path, copy_path)
        with OntologyStore(copy_path) as store:
            counts = store.count_by_type()
            links_before = sum(store.count_links_by_type().values())
            print(f"   실데이터(사본): {counts}, 링크 {links_before}개")
            ok &= check("승격 데이터 존재 (Ordinance·Department·PressRelease)",
                        all(counts.get(t, 0) > 0
                            for t in ("Ordinance", "Department", "PressRelease")))

            ordin = store.find(type="Ordinance", name_like="도시계획", limit=1)
            dept = store.find(type="Department", name_like="주거정비", limit=1)
            if not ordin or not dept:
                return check("다중 홉용 실객체 검색", False, "조례/부서 검색 실패")
            ordin, dept = ordin[0], dept[0]

            dummy = Policy(id="policy:dummy-multihop", name="더미 재개발 사업", year=2026)
            store.upsert_object(dummy)
            store.upsert_links([
                Link(type=LinkType.BASIS_OF, src=ordin.id, dst=dummy.id, method="manual"),
                Link(type=LinkType.MANAGES, src=dept.id, dst=dummy.id, method="manual"),
            ])

            basis = store.neighbors(dummy.id, LinkType.BASIS_OF, direction="in")
            manager = store.neighbors(dummy.id, LinkType.MANAGES, direction="in")
            ok &= check(
                "다중 홉 질의 — 더미 사업의 근거조례·담당부서",
                [o.id for o in basis] == [ordin.id]
                and [o.id for o in manager] == [dept.id],
                f"근거조례='{basis[0].name if basis else '?'}' 담당부서='{manager[0].name if manager else '?'}'",
            )

            g = store.to_networkx()
            ok &= check("NetworkX 그래프 내보내기",
                        g.number_of_nodes() == sum(counts.values()) + 1
                        and g.number_of_edges() == links_before + 2,
                        f"{g.number_of_nodes():,}개 노드 / {g.number_of_edges()}개 엣지")

            store.delete(dummy.id)
            ok &= check("더미 제거 후 원상복구 (사본 내)",
                        store.get(dummy.id) is None
                        and sum(store.count_links_by_type().values()) == links_before
                        and store.count_by_type() == counts)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args()

    print("── 벤치마크 1·2: 빈 그래프 + 더미 왕복 + 링크 규칙 ──")
    ok = verify_empty_roundtrip()
    print("\n── 벤치마크 3: 실데이터 다중 홉 ──")
    ok &= verify_real_multihop(args.db)
    print("\n결과:", "전체 통과 ✅" if ok else "실패 항목 있음 ❌")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
