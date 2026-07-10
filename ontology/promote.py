"""기존 디딤 데이터를 온톨로지 객체로 승격한다 (1단계).

입력: ~/didim/data/index/{seongdong_ordin,seongdong_org,seongdong_boards}.meta.json
출력: data/ontology.db

승격 규칙
- 조례(seongdong_ordin, 644건)      → Ordinance  — 청크를 chunk_no 순으로 합쳐 조문 전문 복원.
  디딤 색인은 슬라이딩 윈도우(약 120자 중첩)로 청크를 잘랐으므로 결합 시 중첩을 제거한다.
- 조직·업무분장(seongdong_org, 41건) → Department — 제목에서 부서명 추출, 업무분장은 조직 단위만
- 보도자료·새소식(seongdong_boards)   → PressRelease — sd/board/{press,news}만. 감사결과(audit)는
  8개 객체 타입에 해당하지 않으므로 제외한다.

개인정보 배제(안전 원칙 2): 보도자료 본문에는 '담당자: 실명(직통번호)' 정형 패턴과
개인 이메일이 섞여 있으므로, 승격 시 이메일·휴대전화 전부와 담당자 실명을 마스킹해
자연인 식별정보가 저장소에 들어오지 않게 한다. 부서·팀 단위 유선 대표번호는 조직
단위 정보라 유지한다(조직·업무분장 데이터와 같은 기준). 원문은 source_url로 열람 가능.

링크(사업↔조례↔예산 등)는 Policy·BudgetItem 객체가 생기는 2~3단계에서 생성한다.

실행: .venv/bin/python -m ontology.promote [--didim-index-dir PATH] [--out DB_PATH]
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .schema import Department, Ordinance, PressRelease, SDObject
from .store import OntologyStore

KST = timezone(timedelta(hours=9))

DEFAULT_INDEX_DIR = Path.home() / "didim" / "data" / "index"
DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "ontology.db"

_ORDINANCE_KINDS = ("조례", "규칙", "훈령", "예규", "규정", "지침")

# 디딤 색인의 청크 중첩은 실측 119~120자 — 여유를 두고 이 범위에서 탐색한다.
_OVERLAP_MAX = 300
_OVERLAP_MIN = 30


def _load_docs(index_dir: Path, basename: str) -> dict[str, list[dict]]:
    """색인 파일을 doc_id → [청크(chunk_no 순)] 로 묶는다."""
    meta = json.loads((index_dir / f"{basename}.meta.json").read_text(encoding="utf-8"))
    docs: dict[str, list[dict]] = {}
    for entry in meta["entries"]:
        docs.setdefault(entry["doc_id"], []).append(entry)
    for chunks in docs.values():
        chunks.sort(key=lambda c: c.get("chunk_no", 0))
    return docs


def _join_text(chunks: list[dict]) -> str:
    """청크를 이어 붙이되 슬라이딩 윈도우 중첩(앞 청크 꼬리 = 뒤 청크 머리)을 제거한다."""
    text = ""
    for c in chunks:
        piece = c.get("text", "")
        if not text:
            text = piece
            continue
        limit = min(len(text), len(piece), _OVERLAP_MAX)
        cut = 0
        for length in range(limit, _OVERLAP_MIN - 1, -1):
            if text[-length:] == piece[:length]:
                cut = length
                break
        text += piece[cut:] if cut else "\n" + piece
    return text.strip()


# ── 개인정보 스크럽 (안전 원칙 2) ────────────────────────────────

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_MOBILE_RE = re.compile(r"\b01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}\b")
# "이승민 주무관" 등 개인 단위 직함 앞의 실명 (부서명은 5자 이상이라 걸리지 않음)
_NAME_TITLE_RE = re.compile(r"(?<![가-힣])([가-힣]{2,4})(\s?)(주무관|사무관|주사보|주사|주임)(?![가-힣])")
# "담당(자): 홍길동" — 뒤에 부서·팀 명칭이 아닌 2~3자 실명
_CONTACT_NAME_RE = re.compile(r"(담당자?\s*[:：]\s*)([가-힣]{2,3})(?![가-힣])")
# "조은진 팀장(070-…)" — 직함 뒤 15자 내 연락처가 붙은 경우만 실명 마스킹
_NAME_TITLE_CONTACT_RE = re.compile(
    r"(?<![가-힣])([가-힣]{2,3})(\s?)(팀장|과장|센터장)(?=[^가-힣]{0,15}(?:0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}|@))"
)
# "마을행정팀 조수영（02-2286-7351）" — 유선번호 괄호가 바로 뒤따르는 실명.
# 조직명 접미(…과/팀/동 등)로 끝나는 낱말은 제외 — 이름에 드문 음절만 고른다.
_NAME_PHONE_RE = re.compile(
    r"(?<![가-힣])([가-힣]{2,3})(?<![과팀동관국단실처청소])"
    r"(\s*[（(]\s*)(?=0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}|[Ee][-–]?mail|이메일|☎)"
)
# "담당 문주현," / "담당 김종수 ☎" / "담당자 유지원 02-…" — '담당' 뒤 실명
_TAKER_NAME_RE = re.compile(
    r"(담당자?\s+)(?!부서|업무|기관|부처)([가-힣]{2,3})"
    r"(?=\s*[,，、(（☎]|\s*0\d{1,2}[-.\s]?\d{3,4})"
)


def scrub_personal_contacts(text: str) -> tuple[str, int]:
    """자연인 식별정보(실명+직통 연락처 결합)를 마스킹한다. 반환: (본문, 치환 수)."""
    n = 0

    def _count(m_repl):
        def inner(m):
            nonlocal n
            n += 1
            return m_repl(m)
        return inner

    text = _EMAIL_RE.sub(_count(lambda m: "(이메일 비공개)"), text)
    text = _MOBILE_RE.sub(_count(lambda m: "(휴대전화 비공개)"), text)
    text = _NAME_TITLE_RE.sub(_count(lambda m: f"○○○{m.group(2)}{m.group(3)}"), text)
    text = _CONTACT_NAME_RE.sub(_count(lambda m: f"{m.group(1)}○○○"), text)
    text = _NAME_TITLE_CONTACT_RE.sub(
        _count(lambda m: f"○○○{m.group(2)}{m.group(3)}"), text
    )
    text = _NAME_PHONE_RE.sub(_count(lambda m: f"○○○{m.group(2)}"), text)
    text = _TAKER_NAME_RE.sub(_count(lambda m: f"{m.group(1)}○○○"), text)
    return text, n


def _year_of(date: str | None) -> int | None:
    if date and len(date) >= 4 and date[:4].isdigit():
        return int(date[:4])
    return None


def promote_ordinances(index_dir: Path) -> list[SDObject]:
    docs = _load_docs(index_dir, "seongdong_ordin")
    out: list[SDObject] = []
    for doc_id, chunks in docs.items():
        head = chunks[0]
        title = head.get("title", "").strip()
        alr_no = doc_id.rsplit("/", 1)[-1]
        kind = next((k for k in _ORDINANCE_KINDS if title.endswith(k)), "기타")
        date = head.get("date")
        out.append(Ordinance(
            id=f"ordinance:{alr_no}",
            name=title,
            kind=kind,
            full_text=_join_text(chunks),
            revision_history=f"최종개정 {date}" if date else None,
            year=_year_of(date),
            source_label=head.get("source_label"),
            source_url=head.get("url"),
        ))
    return out


def promote_departments(index_dir: Path) -> list[SDObject]:
    docs = _load_docs(index_dir, "seongdong_org")
    out: list[SDObject] = []
    for doc_id, chunks in docs.items():
        head = chunks[0]
        title = head.get("title", "").strip()
        name = title.removesuffix(" 조직·업무분장").strip() or title
        key = doc_id.rsplit("/", 1)[-1]
        out.append(Department(
            id=f"department:{key}",
            name=name,
            duties=_join_text(chunks),
            source_label=head.get("source_label"),
            source_url=head.get("url"),
        ))
    return out


_BOARD_SUBTYPES = {"press": "보도자료", "news": "새소식"}


def promote_press_releases(index_dir: Path) -> tuple[list[SDObject], int, int]:
    """보도자료·새소식을 승격한다. 반환: (객체 목록, 제외 문서 수, 개인정보 마스킹 수)."""
    docs = _load_docs(index_dir, "seongdong_boards")
    out: list[SDObject] = []
    skipped = scrubbed = 0
    for doc_id, chunks in docs.items():
        # doc_id 형식: sd/board/{press|news|audit}/<nttNo>
        parts = doc_id.split("/")
        board = parts[2] if len(parts) >= 4 else ""
        if board not in _BOARD_SUBTYPES:
            skipped += 1
            continue
        head = chunks[0]
        body, n_masked = scrub_personal_contacts(_join_text(chunks))
        scrubbed += n_masked
        out.append(PressRelease(
            id=f"press:{board}/{parts[-1]}",
            name=head.get("title", "").strip(),
            subtype=_BOARD_SUBTYPES[board],
            published_at=head.get("date"),
            body=body,
            source_label=head.get("source_label"),
            source_url=head.get("url"),
        ))
    return out, skipped, scrubbed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--didim-index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_DB)
    args = ap.parse_args()

    ordinances = promote_ordinances(args.didim_index_dir)
    departments = promote_departments(args.didim_index_dir)
    presses, skipped, scrubbed = promote_press_releases(args.didim_index_dir)

    with OntologyStore(args.out) as store:
        n = store.upsert_objects(ordinances + departments + presses)
        store.set_meta("promoted_at", datetime.now(KST).isoformat(timespec="seconds"))
        store.set_meta("didim_index_dir", str(args.didim_index_dir))
        counts = store.count_by_type()

    print(f"승격 완료 → {args.out}")
    for t, c in counts.items():
        print(f"  {t:14s} {c:6,d}")
    print(f"  {'합계':14s} {n:6,d}  (감사결과 제외 {skipped}건,"
          f" 개인정보 마스킹 {scrubbed:,}곳)")


if __name__ == "__main__":
    main()
