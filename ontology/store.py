"""온톨로지 SQLite 저장소 — 객체·링크 테이블 + NetworkX 그래프 질의.

가이드라인의 경량 아키텍처 그대로: SQLite(객체·링크 테이블) + NetworkX(인메모리
그래프 질의). 규모가 작으므로 Neo4j 없이 충분하다. 읽기 전용 의미 계층이므로
저장소가 외부 시스템에 쓰기를 하는 일은 없다.

객체 본문은 Pydantic 모델의 JSON 직렬화를 props 컬럼에 통째로 저장하고,
색인이 필요한 id/type/name만 컬럼으로 뽑는다. 로드 시 TYPE_REGISTRY로
원래 타입으로 복원되므로 Pydantic 검증이 왕복 내내 유지된다.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

from .schema import (
    LINK_ENDPOINTS,
    SCHEMA_VERSION,
    TYPE_REGISTRY,
    Link,
    LinkType,
    SDObject,
)

KST = timezone(timedelta(hours=9))

_DDL = """
CREATE TABLE IF NOT EXISTS objects (
    id         TEXT PRIMARY KEY,
    type       TEXT NOT NULL,
    name       TEXT NOT NULL,
    props      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(type);
CREATE INDEX IF NOT EXISTS idx_objects_name ON objects(name);

CREATE TABLE IF NOT EXISTS links (
    id         TEXT PRIMARY KEY,
    type       TEXT NOT NULL,
    src        TEXT NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    dst        TEXT NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    props      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_links_type ON links(type);
CREATE INDEX IF NOT EXISTS idx_links_src ON links(src);
CREATE INDEX IF NOT EXISTS idx_links_dst ON links(dst);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class OntologyStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Streamlit은 스크립트 실행마다 새 스레드를 쓰므로 check_same_thread를 풀고
        # 락으로 연결 접근을 직렬화한다 (sqlite 연결은 스레드 안전이 아님).
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        try:
            # 초기화 쓰기(WAL 전환·스키마·버전 기록)는 읽기 전용 배포 환경
            # (컨테이너의 read-only 마운트 등)에서 실패할 수 있다 — 그 경우
            # 이미 구축된 DB를 조회 전용으로 쓰는 데는 지장이 없으므로 계속한다.
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(_DDL)
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            self.conn.commit()
        except sqlite3.OperationalError:
            self.conn.rollback()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "OntologyStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── 객체 ──────────────────────────────────────────────

    def upsert_object(self, obj: SDObject, commit: bool = True) -> None:
        if type(obj).__name__ not in TYPE_REGISTRY:
            raise ValueError(f"등록되지 않은 객체 타입: {type(obj).__name__}")
        # INSERT OR REPLACE는 SQLite에서 DELETE+INSERT로 처리되어 링크의
        # ON DELETE CASCADE를 발동시킨다(재승격 한 번에 엣지 전멸).
        # ON CONFLICT DO UPDATE는 진짜 UPDATE라 링크가 보존된다.
        with self._lock:
            self.conn.execute(
                "INSERT INTO objects(id, type, name, props, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET"
                " type=excluded.type, name=excluded.name,"
                " props=excluded.props, updated_at=excluded.updated_at",
                (
                    obj.id,
                    type(obj).__name__,
                    obj.name,
                    obj.model_dump_json(),
                    datetime.now(KST).isoformat(timespec="seconds"),
                ),
            )
            if commit:
                self.conn.commit()

    def upsert_objects(self, objs: Iterable[SDObject]) -> int:
        with self._lock:
            n = 0
            try:
                for obj in objs:
                    self.upsert_object(obj, commit=False)
                    n += 1
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return n

    def get(self, obj_id: str) -> Optional[SDObject]:
        with self._lock:
            row = self.conn.execute(
                "SELECT type, props FROM objects WHERE id = ?", (obj_id,)
            ).fetchone()
        if row is None:
            return None
        return TYPE_REGISTRY[row["type"]].model_validate_json(row["props"])

    def find(
        self,
        type: Optional[str] = None,
        name_like: Optional[str] = None,
        limit: int = 50,
    ) -> list[SDObject]:
        sql, params = "SELECT type, props FROM objects WHERE 1=1", []
        if type is not None:
            sql += " AND type = ?"
            params.append(type)
        if name_like is not None:
            escaped = (name_like.replace("\\", "\\\\")
                       .replace("%", "\\%").replace("_", "\\_"))
            sql += " AND name LIKE ? ESCAPE '\\'"
            params.append(f"%{escaped}%")
        sql += " ORDER BY name LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [TYPE_REGISTRY[r["type"]].model_validate_json(r["props"]) for r in rows]

    def delete(self, obj_id: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM objects WHERE id = ?", (obj_id,))
            self.conn.commit()

    # ── 링크 ──────────────────────────────────────────────

    def upsert_link(self, link: Link, commit: bool = True) -> None:
        src_type, dst_type = LINK_ENDPOINTS[link.type]
        with self._lock:
            src_row = self.conn.execute(
                "SELECT type FROM objects WHERE id = ?", (link.src,)
            ).fetchone()
            dst_row = self.conn.execute(
                "SELECT type FROM objects WHERE id = ?", (link.dst,)
            ).fetchone()
            if src_row is None or dst_row is None:
                raise ValueError(f"링크 양끝 객체가 저장소에 없음: {link.src} → {link.dst}")
            if src_row["type"] != src_type or dst_row["type"] != dst_type:
                raise ValueError(
                    f"링크 '{link.type.value}'는 {src_type}→{dst_type}만 허용"
                    f" (실제: {src_row['type']}→{dst_row['type']})"
                )
            self.conn.execute(
                "INSERT INTO links(id, type, src, dst, props, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET"
                " props=excluded.props, updated_at=excluded.updated_at",
                (
                    link.id,
                    link.type.value,
                    link.src,
                    link.dst,
                    link.model_dump_json(exclude={"type", "src", "dst"}),
                    datetime.now(KST).isoformat(timespec="seconds"),
                ),
            )
            if commit:
                self.conn.commit()

    def upsert_links(self, links: Iterable[Link]) -> int:
        with self._lock:
            n = 0
            try:
                for link in links:
                    self.upsert_link(link, commit=False)
                    n += 1
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return n

    def links_of(
        self,
        obj_id: str,
        link_type: Optional[LinkType] = None,
        direction: str = "both",  # out | in | both
        min_confidence: float = 0.0,
    ) -> list[Link]:
        """min_confidence: Claude 생성 링크의 오탐을 소비 시점에 거른다.
        고정밀 경로(질의응답 답변 근거 등)는 0.85 권장 — 표본 검증에서
        오탐 대부분이 0.7(하한 통과치)에 몰려 있었다."""
        clauses, params = [], []
        if direction in ("out", "both"):
            clauses.append("src = ?")
            params.append(obj_id)
        if direction in ("in", "both"):
            clauses.append("dst = ?")
            params.append(obj_id)
        sql = f"SELECT type, src, dst, props FROM links WHERE ({' OR '.join(clauses)})"
        if link_type is not None:
            sql += " AND type = ?"
            params.append(link_type.value)
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        links = []
        for r in rows:
            try:
                lt = LinkType(r["type"])
            except ValueError:
                # DB가 코드보다 새로울 수 있다(배포 프로세스가 옛 모듈을 캐시한
                # 상태에서 DB만 갱신된 경우 등) — 모르는 링크 타입은 조회에서
                # 건너뛰어 전체 질의가 죽지 않게 한다 (전방 호환).
                continue
            links.append(Link(type=lt, src=r["src"], dst=r["dst"],
                              **json.loads(r["props"])))
        return [l for l in links if l.confidence >= min_confidence]

    def neighbors(
        self,
        obj_id: str,
        link_type: Optional[LinkType] = None,
        direction: str = "both",
        min_confidence: float = 0.0,
    ) -> list[SDObject]:
        out = []
        for link in self.links_of(obj_id, link_type, direction, min_confidence):
            other = link.dst if link.src == obj_id else link.src
            obj = self.get(other)
            if obj is not None:
                out.append(obj)
        return out

    # ── 통계·그래프 ───────────────────────────────────────

    def count_by_type(self) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT type, COUNT(*) AS n FROM objects GROUP BY type ORDER BY n DESC"
            ).fetchall()
        return {r["type"]: r["n"] for r in rows}

    def count_links_by_type(self) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT type, COUNT(*) AS n FROM links GROUP BY type ORDER BY n DESC"
            ).fetchall()
        return {r["type"]: r["n"] for r in rows}

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value)
            )
            self.conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def to_networkx(self):
        """전체 온톨로지를 NetworkX 방향 그래프로 — 다중 홉 질의(GraphRAG 4단계)용.

        노드 속성은 name/type만 싣는다(본문은 필요할 때 store.get으로).
        """
        import networkx as nx

        g = nx.MultiDiGraph()
        with self._lock:
            for r in self.conn.execute("SELECT id, type, name FROM objects"):
                g.add_node(r["id"], type=r["type"], name=r["name"])
            for r in self.conn.execute("SELECT type, src, dst FROM links"):
                g.add_edge(r["src"], r["dst"], type=r["type"])
        return g
