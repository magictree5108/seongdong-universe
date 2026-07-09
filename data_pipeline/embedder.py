"""문자 n-gram 해싱 TF-IDF 임베더 — 순수 로컬, 모델 다운로드 없음.

디딤(Didim) 프로젝트의 색인 임베더와 같은 방식(해싱 기반 n-gram TF-IDF)을
독립적으로 재구현했다. NEO 성동은 didim 백엔드에 의존하지 않는 완전히
분리된 프로젝트이므로, 빌드 시점(build.py)과 검색 실행 시점(app.py) 모두
이 모듈 하나만으로 동일한 임베딩 공간을 재현한다.
"""
import hashlib
import re

import numpy as np

DIM = 1024


def _ngrams(text: str):
    text = re.sub(r"\s+", " ", text.lower())
    for n in (2, 3):
        for i in range(len(text) - n + 1):
            yield text[i:i + n]


def _bucket(gram: str) -> int:
    h = hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "little") % DIM


def counts(texts: list[str]) -> np.ndarray:
    """sublinear TF 행렬 (정규화 전)."""
    mat = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for gram in _ngrams(text):
            mat[row, _bucket(gram)] += 1.0
    return np.log1p(mat)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def fit_idf(texts: list[str]) -> np.ndarray:
    """코퍼스 전체의 문서빈도로 IDF를 집계한다 (빌드 시 1회)."""
    c = counts(texts)
    df = (c > 0).sum(axis=0).astype(np.float32)
    return (np.log((len(texts) + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)


def embed(texts: list[str], idf: np.ndarray) -> np.ndarray:
    """빌드된 idf로 텍스트를 임베딩한다 (검색 질의 임베딩에도 동일하게 사용)."""
    return l2_normalize(counts(texts) * idf)
