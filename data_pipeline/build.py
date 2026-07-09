"""디딤(Didim) 색인에서 NEO 성동 데이터(노드·엣지·임베딩)를 빌드한다.

입력: ~/didim/data/index/{audit_cases,seongdong_notices,seongdong_ordin}.meta.json
출력: data/nodes.json, data/edges.json, data/embeddings.npy, data/idf.npy, data/meta.json

빌드된 결과물은 저장소에 함께 커밋되므로, 배포 시(Streamlit Community Cloud 등)
디딤 백엔드나 원본 데이터 없이도 NEO 성동 앱이 완전히 독립적으로 동작한다.

실행: python data_pipeline/build.py [--didim-index-dir PATH] [--out DATA_DIR]
"""
import argparse
import json
import re
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
]

MAX_CHUNKS_PER_DOC = 2       # 문서당 결합할 청크 수 (임베딩·본문용)
SNIPPET_LEN = 260            # 화면 표시용 발췌 길이
KNN_K = 4                    # 노드당 최대 이웃 수
SIM_FLOOR = 0.18             # 이 아래 유사도는 애초에 이웃 후보에서 제외
MAX_EDGES = 7000             # 렌더 성능을 위한 엣지 총량 상한


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def load_docs(index_dir: Path) -> list[dict]:
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


def build_knn_edges(vectors: np.ndarray) -> list[tuple[int, int, float]]:
    """코사인 유사도 기반 k-최근접 이웃 그래프. 유사도 임계값을 자동으로
    끌어올려 렌더 가능한 엣지 수(MAX_EDGES) 이하로 맞춘다."""
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
        if len(pairs) <= MAX_EDGES or floor >= 0.6:
            break
        floor += 0.05
    edges = [(i, j, round(w, 3)) for (i, j), w in pairs.items()]
    print(f"엣지 {len(edges)}개 (유사도 임계값 {floor:.2f})")
    return edges


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

    degree = np.zeros(len(docs), dtype=np.int32)
    for i, j, _w in edges:
        degree[i] += 1
        degree[j] += 1

    nodes = []
    for i, d in enumerate(docs):
        nodes.append({
            "id": i,
            "doc_id": d["doc_id"],
            "title": d["title"] or "(제목 없음)",
            "category": d["category"],
            "category_label": d["category_label"],
            "source_label": d["source_label"],
            "url": d["url"],
            "date": d["date"],
            "snippet": (d["text"][:SNIPPET_LEN] + "…") if len(d["text"]) > SNIPPET_LEN else d["text"],
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "z": round(float(coords[i, 2]), 4),
            "degree": int(degree[i]),
        })

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "nodes.json").write_text(
        json.dumps(nodes, ensure_ascii=False), encoding="utf-8")
    (args.out / "edges.json").write_text(
        json.dumps(edges, ensure_ascii=False), encoding="utf-8")
    np.save(args.out / "embeddings.npy", vectors.astype(np.float32))
    np.save(args.out / "idf.npy", idf.astype(np.float32))
    (args.out / "meta.json").write_text(json.dumps({
        "built_at": datetime.now(KST).isoformat(timespec="seconds"),
        "embedder": "hash-ngram-idf-v2 (neo-seongdong 독립 재구현)",
        "dim": embedder.DIM,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "categories": {slug: sum(1 for d in docs if d["category"] == slug)
                       for _l, slug, _b in SOURCES},
        "didim_index_built_at": source_built_at,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"완료: 노드 {len(nodes)}개, 엣지 {len(edges)}개 → {args.out}")


if __name__ == "__main__":
    main()
