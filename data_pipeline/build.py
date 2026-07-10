"""디딤(Didim) 색인에서 NEO 성동 데이터(노드·엣지·온톨로지·임베딩)를 빌드한다.

입력: ~/didim/data/index/{audit_cases,seongdong_notices,seongdong_ordin}.meta.json
출력: data/nodes.json, data/edges.json, data/embeddings.npy, data/idf.npy, data/meta.json

온톨로지 레이어 (팔란티어 Gotham의 객체 모델에서 영감):
문서에서 규칙 기반으로 개체(entity)를 추출해 별도 노드로 승격한다.
- dept  담당 부서   — 고시공고 제목의 "(담당: ○○과)" 패턴
- law   법령 참조   — 본문의 「…법/조례/규칙」 인용
- place 성동구 동네 — 행정동/법정동 이름 매칭
개체는 연결된 문서들의 중심(centroid)에 배치되고, 문서-개체 엣지는
타입을 가진다. LLM 없이 결정적(deterministic)으로 추출된다.

빌드된 결과물은 저장소에 함께 커밋되므로, 배포 시(Streamlit Community Cloud 등)
디딤 백엔드나 원본 데이터 없이도 NEO 성동 앱이 완전히 독립적으로 동작한다.

실행: python data_pipeline/build.py [--didim-index-dir PATH] [--out DATA_DIR]
"""
import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import embedder

KST = timezone(timedelta(hours=9))

# 색인 파일 → (카테고리 표시명, 카테고리 slug, 색인 파일 basename)
SOURCES = [
    ("사전컨설팅·면책 선례 (감사원·자체감사기구)", "consulting", "audit_cases"),
    ("성동구 고시공고", "notice", "seongdong_notices"),
    ("성동구 자치법규", "ordinance", "seongdong_ordin"),
    ("성동구 조직·업무분장", "org", "seongdong_org"),
    ("성동구 보도·소식·감사결과", "news", "seongdong_boards"),
]

MAX_CHUNKS_PER_DOC = 2       # 문서당 결합할 청크 수 (임베딩·본문·개체추출용)
SNIPPET_LEN = 460            # 화면 표시용 발췌 길이
GATE_LEN = 800               # 어휘 게이트용 본문 길이 (검색어 실존 확인)
KNN_K = 4                    # 문서 노드당 최대 이웃 수
SIM_FLOOR = 0.18             # 이 아래 유사도는 애초에 이웃 후보에서 제외
MAX_SIM_EDGES = 7000         # 유사도 엣지 총량 상한 (렌더 성능)

# ── 온톨로지 추출 규칙 ──────────────────────────────────────────
MIN_ENTITY_LINKS = 3         # 이보다 적게 연결된 개체는 버린다 (잡음 억제)
MAX_ENTITIES = 120           # 화면 밀도를 위한 개체 수 상한 (연결 수 상위)
MAX_ENTITY_VIZ_EDGES = 40    # 개체당 그리는 엣지 상한 (전체 연결은 노드에 보존)

# ── 온톨로지 Policy(사업) 계층 ──────────────────────────────────
# ontology/ 패키지가 만든 data/ontology.db의 사업·링크를 우주 노드로 승격한다.
# 조례·보도·부서 문서 노드와 원천이 같아 ID가 정확히 매핑된다.
ONTOLOGY_DB = Path(__file__).resolve().parent.parent / "data" / "ontology.db"
ONTO_MIN_CONF = 0.85         # Claude 생성 링크의 고정밀 소비 기준 (표본 검증 근거)

_DEPT_RE = re.compile(r"\(담당\s*:\s*([^)]{2,20})\)")
_LAW_RE = re.compile(r"「\s*([^」]{2,40}?)\s*」")
_LAW_SUFFIXES = ("법", "법률", "조례", "규칙", "규정", "훈령", "예규", "지침", "영")
_PLACES = [
    "성수동", "왕십리", "마장동", "사근동", "행당동", "응봉동",
    "금호동", "옥수동", "송정동", "용답동", "도선동", "홍익동",
]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _year_of(date: str | None) -> int | None:
    if date and re.match(r"^\d{4}", date):
        return int(date[:4])
    return None


def extract_entities(doc: dict) -> list[tuple[str, str]]:
    """문서에서 (개체타입, 개체이름) 목록을 규칙 기반으로 추출한다."""
    found: set[tuple[str, str]] = set()
    title, text = doc["title"], doc["text"]

    for m in _DEPT_RE.finditer(title):
        name = _clean(m.group(1))
        if name.endswith(("과", "국", "소", "센터", "단")):
            found.add(("dept", name))

    # 조직·업무분장 문서는 그 자체가 부서 문서다 — 제목에서 부서 개체를 잇는다
    if doc["category"] == "org":
        m = re.match(r"^(\S+)\s+조직·업무분장", title)
        if m and m.group(1).endswith(("과", "국", "소", "센터", "단", "담당관", "동", "실")):
            found.add(("dept", m.group(1)))

    for m in _LAW_RE.finditer(f"{title} {text}"):
        name = _clean(m.group(1))
        if 2 <= len(name) <= 40 and name.endswith(_LAW_SUFFIXES):
            found.add(("law", name))

    haystack = f"{title} {text}"
    for place in _PLACES:
        if place in haystack:
            found.add(("place", place))
    return sorted(found)


def load_docs(index_dir: Path) -> tuple[list[dict], dict]:
    """세 색인의 청크를 문서 단위로 합쳐 (title, text, category, ...) 목록을 만든다."""
    docs: dict[str, dict] = {}
    source_built_at = {}
    for label, slug, basename in SOURCES:
        meta_path = index_dir / f"{basename}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        source_built_at[slug] = meta.get("built_at")
        by_doc: dict[str, list[dict]] = {}
        for e in meta["entries"]:
            by_doc.setdefault(e["doc_id"], []).append(e)
        for doc_id, chunks in by_doc.items():
            chunks.sort(key=lambda c: c["chunk_no"])
            head = chunks[0]
            combined = " ".join(_clean(c["text"]) for c in chunks[:MAX_CHUNKS_PER_DOC])
            docs[f"{slug}:{doc_id}"] = {
                "doc_id": doc_id,
                "title": _clean(head["title"]),
                "category": slug,
                "category_label": label,
                "source_label": head["source_label"],
                "url": head.get("url"),
                "date": head.get("date"),
                "text": combined,
            }
    print(f"문서 {len(docs)}건 로드 (원본 색인 빌드 시각: {source_built_at})")
    return list(docs.values()), source_built_at


def pca_3d(vectors: np.ndarray, spread: float = 9.0) -> np.ndarray:
    """중심화 후 SVD로 상위 3개 주성분에 투영 (외부 의존성 없는 순수 numpy PCA)."""
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :3] * s[:3]
    scale = spread / (np.abs(coords).std() + 1e-8)
    return (coords * scale).astype(np.float32)


def build_knn_edges(vectors: np.ndarray) -> list[list]:
    """코사인 유사도 기반 k-최근접 이웃 그래프. 유사도 임계값을 자동으로
    끌어올려 렌더 가능한 엣지 수(MAX_SIM_EDGES) 근처로 맞춘다."""
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1.0)
    n = vectors.shape[0]
    top_idx = np.argpartition(-sims, KNN_K, axis=1)[:, :KNN_K]

    floor = SIM_FLOOR
    while True:
        pairs: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in top_idx[i]:
                score = float(sims[i, j])
                if score < floor:
                    continue
                key = (i, int(j)) if i < j else (int(j), i)
                if key not in pairs or score > pairs[key]:
                    pairs[key] = score
        if len(pairs) <= MAX_SIM_EDGES or floor >= 0.6:
            break
        floor += 0.05
    edges = [[i, j, round(w, 3), "sim"] for (i, j), w in pairs.items()]
    print(f"유사도 엣지 {len(edges)}개 (임계값 {floor:.2f})")
    return edges


def build_ontology(docs: list[dict], coords: np.ndarray,
                   vectors: np.ndarray) -> tuple[list[dict], list[list], np.ndarray, np.ndarray]:
    """문서에서 개체를 추출해 개체 노드·타입 엣지·개체 좌표/벡터를 만든다."""
    linked: dict[tuple[str, str], list[int]] = defaultdict(list)
    for di, doc in enumerate(docs):
        for etype, name in extract_entities(doc):
            linked[(etype, name)].append(di)

    kept = {k: v for k, v in linked.items() if len(v) >= MIN_ENTITY_LINKS}
    # 부서 개체는 온톨로지의 축이므로 캡과 무관하게 전부 채택하고,
    # 나머지(법령·동네)는 연결 수 상위로 캡을 채운다
    dept_items = [kv for kv in kept.items() if kv[0][0] == "dept"]
    other_items = sorted((kv for kv in kept.items() if kv[0][0] != "dept"),
                         key=lambda kv: -len(kv[1]))
    ranked = dept_items + other_items[:max(0, MAX_ENTITIES - len(dept_items))]
    print(f"개체 후보 {len(linked)}종 → 채택 {len(ranked)}종 "
          f"(부서 {len(dept_items)} 전부 + 기타 상위, 연결 {MIN_ENTITY_LINKS}건 이상)")

    etype_labels = {"dept": "담당 부서", "law": "법령·자치법규 참조", "place": "성동구 동네"}
    entity_nodes: list[dict] = []
    entity_edges: list[list] = []
    ecoords: list[np.ndarray] = []
    evecs: list[np.ndarray] = []
    for (etype, name), doc_idx in ranked:
        eid = len(docs) + len(entity_nodes)
        member = np.array(doc_idx)
        centroid = coords[member].mean(axis=0)
        # 같은 자리에 겹치지 않게 이름 해시 기반의 결정적 오프셋을 준다
        h = int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(), "little")
        ang, lift = (h % 628) / 100.0, ((h >> 12) % 200 - 100) / 120.0
        centroid = centroid + np.array([math.cos(ang), math.sin(ang), lift],
                                       dtype=np.float32) * 0.9
        vec = vectors[member].mean(axis=0)
        vec = vec / (np.linalg.norm(vec) + 1e-9)

        entity_nodes.append({
            "id": eid,
            "doc_id": f"entity/{etype}/{name}",
            "title": name,
            "kind": "entity",
            "etype": etype,
            "category": "entity",
            "category_label": etype_labels[etype],
            "source_label": f"온톨로지 개체 — {etype_labels[etype]}",
            "url": None,
            "date": None,
            "year": None,
            "snippet": f"규칙 기반 추출 개체. 문서 {len(doc_idx)}건과 연결.",
            "gate_text": name.lower(),
            "x": round(float(centroid[0]), 4),
            "y": round(float(centroid[1]), 4),
            "z": round(float(centroid[2]), 4),
            "degree": len(doc_idx),
            "links": doc_idx,        # 전체 연결 (Object 360용)
        })
        ecoords.append(centroid)
        evecs.append(vec)
        # 렌더용 엣지는 개체당 상한을 둔다 (허브 개체의 시각적 포화 방지)
        step = max(1, len(doc_idx) // MAX_ENTITY_VIZ_EDGES)
        for di in doc_idx[::step][:MAX_ENTITY_VIZ_EDGES]:
            entity_edges.append([di, eid, 1.0, etype])

    ecoords_arr = (np.array(ecoords, dtype=np.float32)
                   if ecoords else np.zeros((0, 3), dtype=np.float32))
    evecs_arr = (np.array(evecs, dtype=np.float32)
                 if evecs else np.zeros((0, vectors.shape[1]), dtype=np.float32))
    print(f"개체 엣지 {len(entity_edges)}개 (렌더용, 개체당 최대 {MAX_ENTITY_VIZ_EDGES})")
    return entity_nodes, entity_edges, ecoords_arr, evecs_arr


def _fmt_amount(won: int) -> str:
    if won >= 100_000_000:
        return f"{won / 100_000_000:.1f}억"
    return f"{won / 10_000:,.0f}만"


def build_policy_layer(docs: list[dict], coords: np.ndarray, idf: np.ndarray,
                       first_id: int) -> tuple[list[dict], list[list], np.ndarray]:
    """온톨로지 DB의 Policy(사업)를 우주 노드로 승격한다.

    배치: 근거 조례·언급 보도·담당 부서 문서 노드들의 중심점 (개체와 동일 방식).
    엣지: 담당·근거·언급 링크(확신도 ONTO_MIN_CONF 이상)를 그대로 시각화.
    스니펫: 연도별 예산현액·집행률 요약 — 클릭 카드가 곧 미니 사업 360이 된다.
    """
    import sqlite3
    if not ONTOLOGY_DB.exists():
        print("온톨로지 DB 없음 — Policy 계층 생략")
        return [], [], np.zeros((0, embedder.DIM), dtype=np.float32)

    conn = sqlite3.connect(str(ONTOLOGY_DB))
    conn.row_factory = sqlite3.Row
    doc_idx = {d["doc_id"]: i for i, d in enumerate(docs)}

    def _map(oid: str) -> int | None:
        kind, _, rest = oid.partition(":")
        if kind == "ordinance":
            return doc_idx.get(f"sd/ordin/{rest}")
        if kind == "press":
            return doc_idx.get(f"sd/board/{rest}")
        if kind == "department":
            return doc_idx.get(f"sd/org/{rest}")
        return None

    policies = {}
    for r in conn.execute("SELECT id, props FROM objects WHERE type='Policy'"):
        policies[r["id"]] = json.loads(r["props"])

    anchors: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for r in conn.execute(
            "SELECT type, src, dst, props FROM links WHERE type IN ('담당','근거','언급')"):
        conf = json.loads(r["props"]).get("confidence", 1.0)
        if conf < ONTO_MIN_CONF:
            continue
        di = _map(r["src"])
        if di is not None and r["dst"] in policies:
            anchors[r["dst"]].append((di, conf, r["type"]))

    # 연도별 예산 요약 (집행 링크 → BudgetItem)
    budget_by_policy: dict[str, dict[int, list[int]]] = defaultdict(dict)
    budget_rows = {r["id"]: json.loads(r["props"]) for r in conn.execute(
        "SELECT id, props FROM objects WHERE type='BudgetItem'")}
    for r in conn.execute("SELECT src, dst FROM links WHERE type='집행'"):
        b = budget_rows.get(r["dst"])
        if b is None or not b.get("year"):
            continue
        acc = budget_by_policy[r["src"]].setdefault(b["year"], [0, 0])
        acc[0] += b.get("budget_current") or 0
        acc[1] += b.get("expenditure") or 0
    conn.close()

    texts, metas = [], []
    for pid, p in policies.items():
        years = budget_by_policy.get(pid, {})
        segs = []
        for y in sorted(years):
            bdg, ep = years[y]
            pct = f" 집행 {ep / bdg * 100:.0f}%" if bdg and ep else ""
            segs.append(f"{y}년 {_fmt_amount(bdg)}원{pct}")
        summary = " → ".join(segs) or "예산 정보 없음"
        dept = p.get("department") or "부서 미정합"
        basis = f' 근거: {p["basis_ordinance"]}.' if p.get("basis_ordinance") else ""
        snippet = (f'{p.get("field") or "분야 미상"} · 담당 {dept}. '
                   f'예산 {summary}.{basis} (지방재정365 세부사업, 조회일 기준)')
        texts.append(f'{p["name"]} {dept} {p.get("field") or ""} 사업 예산')
        metas.append((pid, p, snippet))

    pvecs = embedder.embed(texts, idf) if texts else \
        np.zeros((0, embedder.DIM), dtype=np.float32)

    policy_nodes, policy_edges = [], []
    for k, (pid, p, snippet) in enumerate(metas):
        gid = first_id + k
        anc = anchors.get(pid, [])
        if anc:
            centroid = coords[[a[0] for a in anc]].mean(axis=0)
        else:
            centroid = np.zeros(3, dtype=np.float32)
        h = int.from_bytes(hashlib.blake2b(pid.encode(), digest_size=4).digest(), "little")
        ang, lift = (h % 628) / 100.0, ((h >> 12) % 200 - 100) / 100.0
        centroid = centroid + np.array(
            [math.cos(ang), math.sin(ang), lift], dtype=np.float32) * (1.1 if anc else 7.5)
        policy_nodes.append({
            "id": gid,
            "doc_id": pid,
            "title": p["name"],
            "kind": "policy",
            "etype": None,
            "category": "policy",
            "category_label": "정책·사업 (예산 온톨로지)",
            "source_label": f'지방재정365 · {p.get("department") or "부서 미정합"}',
            "url": p.get("source_url"),
            "date": str(p.get("year") or ""),
            "year": p.get("year"),
            "snippet": snippet,
            "gate_text": texts[k].lower(),
            "x": round(float(centroid[0]), 4),
            "y": round(float(centroid[1]), 4),
            "z": round(float(centroid[2]), 4),
            "degree": len(anc),
            "links": [a[0] for a in anc],
        })
        for di, conf, _lt in anc:
            policy_edges.append([di, gid, round(conf, 2), "onto"])

    n_anchored = sum(1 for n in policy_nodes if n["degree"])
    print(f"사업 노드 {len(policy_nodes)}개 (문서 연결 {n_anchored}개), "
          f"온톨로지 엣지 {len(policy_edges)}개")
    return policy_nodes, policy_edges, pvecs.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--didim-index-dir", type=Path,
                    default=Path.home() / "didim" / "data" / "index")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = ap.parse_args()

    docs, source_built_at = load_docs(args.didim_index_dir)
    texts = [f'{d["title"]} {d["text"]}' for d in docs]

    print("IDF 집계 + 임베딩 계산 중...")
    idf = embedder.fit_idf(texts)
    vectors = embedder.embed(texts, idf)

    print("3D 좌표 계산(PCA) 중...")
    coords = pca_3d(vectors)

    print("의미 연결망(k-NN) 계산 중...")
    edges = build_knn_edges(vectors)

    print("온톨로지 개체 추출 중...")
    entity_nodes, entity_edges, ecoords, evecs = build_ontology(docs, coords, vectors)
    edges += entity_edges

    print("온톨로지 사업(Policy) 계층 승격 중...")
    policy_nodes, policy_edges, pvecs = build_policy_layer(
        docs, coords, idf, first_id=len(docs) + len(entity_nodes))
    edges += policy_edges

    degree = np.zeros(len(docs), dtype=np.int32)
    for i, j, _w, t in edges:
        if t == "sim":
            degree[i] += 1
            degree[j] += 1

    nodes = []
    for i, d in enumerate(docs):
        nodes.append({
            "id": i,
            "doc_id": d["doc_id"],
            "title": d["title"] or "(제목 없음)",
            "kind": "doc",
            "etype": None,
            "category": d["category"],
            "category_label": d["category_label"],
            "source_label": d["source_label"],
            "url": d["url"],
            "date": d["date"],
            "year": _year_of(d["date"]),
            "snippet": (d["text"][:SNIPPET_LEN] + "…") if len(d["text"]) > SNIPPET_LEN else d["text"],
            "gate_text": f'{d["title"]} {d["text"][:GATE_LEN]}'.lower(),
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "z": round(float(coords[i, 2]), 4),
            "degree": int(degree[i]),
            "links": [],
        })
    nodes += entity_nodes
    nodes += policy_nodes

    all_vectors = np.vstack([vectors, evecs, pvecs]).astype(np.float32)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "nodes.json").write_text(
        json.dumps(nodes, ensure_ascii=False), encoding="utf-8")
    (args.out / "edges.json").write_text(
        json.dumps(edges, ensure_ascii=False), encoding="utf-8")
    np.save(args.out / "embeddings.npy", all_vectors)
    np.save(args.out / "idf.npy", idf.astype(np.float32))

    entity_counts = defaultdict(int)
    for n in entity_nodes:
        entity_counts[n["etype"]] += 1
    (args.out / "meta.json").write_text(json.dumps({
        "built_at": datetime.now(KST).isoformat(timespec="seconds"),
        "embedder": "hash-ngram-idf-v2 (neo-seongdong 독립 재구현)",
        "dim": embedder.DIM,
        "total_nodes": len(nodes),
        "total_docs": len(docs),
        "total_entities": len(entity_nodes),
        "total_policies": len(policy_nodes),
        "total_edges": len(edges),
        "categories": {slug: sum(1 for d in docs if d["category"] == slug)
                       for _l, slug, _b in SOURCES},
        "entity_types": dict(entity_counts),
        "didim_index_built_at": source_built_at,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    years = [n["year"] for n in nodes if n["year"]]
    print(f"완료: 문서 {len(docs)} + 개체 {len(entity_nodes)} + 사업 {len(policy_nodes)}"
          f" = 노드 {len(nodes)}개, 엣지 {len(edges)}개, "
          f"연도 범위 {min(years)}–{max(years)} → {args.out}")


if __name__ == "__main__":
    main()
