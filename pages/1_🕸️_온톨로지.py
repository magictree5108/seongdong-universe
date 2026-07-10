"""성동 유니버스 온톨로지 — 질의응답(GraphRAG) + 사업 360도 뷰.

메인 페이지(성동 UNIVERSE)가 '문서 우주'라면 이 페이지는 그 아래의
'의미 계층'이다: 사업·예산·조례·부서·보도가 1급 객체로 연결된 그래프를
자연어로 질의하고, 사업 하나를 골라 연결된 모든 것을 본다.
"""
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ontology.graphrag import answer_question  # noqa: E402
from ontology.schema import LinkType  # noqa: E402
from ontology.store import OntologyStore  # noqa: E402

DB_PATH = ROOT / "data" / "ontology.db"

st.set_page_config(page_title="성동 온톨로지", page_icon="🕸️",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    "<div style='background:#1a1d29;border:1px solid #3a3f55;border-radius:8px;"
    "padding:6px 14px;font-size:0.8rem;color:#9aa3c0;margin-bottom:14px'>"
    "🕸️ <b>성동 온톨로지</b> — 공개 행정데이터(조례·예산·보도·조직)를 객체·링크로 "
    "연결한 <b>개인 학습용 프로토타입</b>입니다. 성동구 공식 시스템이 아니며 "
    "적법성·정확성을 보증하지 않습니다. 자동 추출 링크에는 확신도가 표시됩니다."
    "</div>", unsafe_allow_html=True)


def _api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:  # noqa: BLE001 — secrets.toml 미존재 등
        return None


@st.cache_resource
def _store() -> OntologyStore:
    return OntologyStore(DB_PATH)


@st.cache_data
def _policy_names() -> list[str]:
    store = _store()
    rows = store.conn.execute(
        "SELECT name FROM objects WHERE type='Policy' ORDER BY name").fetchall()
    return [r["name"] for r in rows]


store = _store()
counts = store.count_by_type()
link_counts = store.count_links_by_type()

st.title("🕸️ 성동 온톨로지")
st.caption(
    f"객체 {sum(counts.values()):,}개 (사업 {counts.get('Policy', 0):,} · "
    f"예산 {counts.get('BudgetItem', 0):,} · 보도 {counts.get('PressRelease', 0):,} · "
    f"조례 {counts.get('Ordinance', 0):,} · 부서 {counts.get('Department', 0)}) · "
    f"링크 {sum(link_counts.values()):,}개 "
    f"(집행 {link_counts.get('집행', 0):,} · 담당 {link_counts.get('담당', 0):,} · "
    f"언급 {link_counts.get('언급', 0):,} · 근거 {link_counts.get('근거', 0):,})")

# st.tabs는 rerun마다 첫 탭으로 리셋되어 탐색기의 피벗 이동이 불가능하다 —
# key로 상태가 유지되는 segmented_control을 쓴다.
VIEW_QA, VIEW_360, VIEW_EXPLORE = "💬 질의응답 (GraphRAG)", "🎯 사업 360°", "🧭 탐색기"
view = st.segmented_control(
    "보기", [VIEW_QA, VIEW_360, VIEW_EXPLORE],
    default=VIEW_QA, key="onto_view", label_visibility="collapsed")

# ── 질의응답 ─────────────────────────────────────────────────────

if view == VIEW_QA:
    EXAMPLES = [
        "마을버스 사업의 근거 조례와 올해 예산, 담당부서는?",
        "성동형 스마트쉼터는 어느 부서가 맡고 예산이 얼마나 되나?",
        "1인 가구 지원과 관련된 조례와 사업을 알려줘",
        "주택정비 사업 예산이 최근 몇 년간 어떻게 변했어?",
    ]
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex[:22] + "…", key=f"ex_{ex[:8]}", use_container_width=True):
            st.session_state["onto_q"] = ex

    question = st.text_input(
        "질문", key="onto_q",
        placeholder="예) ○○ 사업의 근거 조례와 올해 예산은? — 그래프에 있는 사실만, 근거 ID와 함께 답합니다")

    if question:
        key = _api_key()
        if not key:
            st.error("ANTHROPIC_API_KEY가 설정되어 있지 않습니다 (Secrets 확인).")
        else:
            with st.spinner("서브그래프를 찾아 근거를 다는 중…"):
                try:
                    result = answer_question(question, db_path=DB_PATH, api_key=key)
                except Exception as e:  # noqa: BLE001
                    st.error(f"질의 실패: {e}")
                    result = None
            if result:
                st.markdown(result["answer"])
                if result["sources"]:
                    st.markdown("##### 근거 원문")
                    for s in result["sources"]:
                        label = f"**{s['name']}** ({s['type']}) · `{s['id']}`"
                        if s["url"]:
                            st.markdown(f"- {label} — [원문 열기]({s['url']})")
                        else:
                            st.markdown(f"- {label}")
                with st.expander(
                        f"서브그래프 — 객체 {result['stats']['objects']}개 · "
                        f"링크 {result['stats']['links']}개 (시드: "
                        + ", ".join(s["name"] for s in result["seeds"]) + ")"):
                    st.json(result["seeds"])

# ── 사업 360° ────────────────────────────────────────────────────

if view == VIEW_360:
    picked = st.selectbox("사업 선택", _policy_names(), index=None,
                          placeholder="사업명을 검색하세요 (1,813개)")
    if picked:
        pol = store.find(type="Policy", name_like=picked, limit=1)
        pol = pol[0] if pol else None
        if pol:
            left, right = st.columns([1, 1])
            with left:
                st.subheader(pol.name)
                st.markdown(
                    f"- **분야**: {pol.field or '—'}\n"
                    f"- **담당부서**: {pol.department or '(미정합)'}\n"
                    f"- **근거조례**: {pol.basis_ordinance or '(연결 없음)'}")
                budgets = sorted(
                    store.neighbors(pol.id, LinkType.EXECUTES, direction="out"),
                    key=lambda b: b.year or 0)
                if budgets:
                    st.markdown("##### 연도별 예산 (지방재정365, 조회일 기준)")
                    rows = []
                    for b in budgets:
                        pct = (b.expenditure / b.budget_current * 100
                               if b.budget_current and b.expenditure else None)
                        rows.append({
                            "연도": b.year, "회계": b.account,
                            "예산현액(원)": b.budget_current,
                            "지출액(원)": b.expenditure,
                            "집행률": f"{pct:.0f}%" if pct is not None else "—",
                        })
                    st.dataframe(rows, use_container_width=True, hide_index=True)
            with right:
                basis = store.links_of(pol.id, LinkType.BASIS_OF, "in",
                                       min_confidence=0.85)
                if basis:
                    st.markdown("##### 근거 조례")
                    for l in basis:
                        o = store.get(l.src)
                        st.markdown(f"- [{o.name}]({o.source_url}) — 확신도 {l.confidence}")
                mentions = store.links_of(pol.id, LinkType.MENTIONS, "in",
                                          min_confidence=0.85)
                if mentions:
                    st.markdown(f"##### 이 사업을 다룬 보도·소식 ({len(mentions)}건)")
                    for l in sorted(mentions, key=lambda x: -(x.confidence or 0))[:8]:
                        pr = store.get(l.src)
                        date = f" ({pr.published_at})" if pr.published_at else ""
                        st.markdown(f"- [{pr.name}]({pr.source_url}){date}")
                if not basis and not mentions:
                    st.info("연결된 조례·보도가 아직 없습니다 (확신도 0.85 이상 기준).")

# ── 탐색기 (객체 → 링크 이웃 피벗) ───────────────────────────────

TYPE_ICON = {"Policy": "🎯", "BudgetItem": "💰", "Ordinance": "📜",
             "PressRelease": "📰", "Department": "🏛️"}

if view == VIEW_EXPLORE:
    st.caption("객체 하나를 잡고 연결을 타고 이동하는 피벗 탐색 — 팔란티어의 그래프 수사 흐름입니다.")

    query = st.text_input("객체 검색", key="explorer_q",
                          placeholder="이름 일부를 입력하세요 (사업·조례·부서·보도 전체 검색)")
    if query:
        hits = store.find(name_like=query, limit=8)
        if not hits:
            st.info("일치하는 객체가 없습니다.")
        for o in hits:
            label = (f"{TYPE_ICON.get(type(o).__name__, '·')} {o.name[:46]}"
                     f"  ({type(o).__name__})")
            if st.button(label, key=f"hit_{o.id}", use_container_width=True):
                st.session_state["explorer_id"] = o.id
                st.rerun()

    cur_id = st.session_state.get("explorer_id")
    if cur_id:
        obj = store.get(cur_id)
        if obj is None:
            st.warning("객체를 찾을 수 없습니다.")
        else:
            tname = type(obj).__name__
            st.markdown(f"### {TYPE_ICON.get(tname, '·')} {obj.name}")
            meta_bits = [f"`{obj.id}`", tname]
            if getattr(obj, "year", None):
                meta_bits.append(f"연도 {obj.year}")
            if getattr(obj, "department", None):
                meta_bits.append(f"부서 {obj.department}")
            if getattr(obj, "field", None):
                meta_bits.append(f"분야 {obj.field}")
            if getattr(obj, "budget_current", None):
                meta_bits.append(f"예산현액 {obj.budget_current:,}원")
            st.markdown(" · ".join(meta_bits))
            if obj.source_url:
                st.markdown(f"[원문 열기]({obj.source_url})")
            body = (getattr(obj, "body", None) or getattr(obj, "full_text", None)
                    or getattr(obj, "duties", None))
            if body:
                with st.expander("본문 발췌"):
                    st.text(body[:1200])

            links = store.links_of(obj.id, direction="both", min_confidence=0.85)
            if not links:
                st.info("확신도 0.85 이상으로 연결된 이웃이 없습니다.")
            by_type: dict[str, list] = {}
            for l in links:
                by_type.setdefault(l.type.value, []).append(l)
            for lt, ls in sorted(by_type.items(), key=lambda x: -len(x[1])):
                st.markdown(f"##### {lt} ({len(ls)})")
                for l in ls[:15]:
                    other_id = l.dst if l.src == obj.id else l.src
                    other = store.get(other_id)
                    if other is None:
                        continue
                    oname = type(other).__name__
                    arrow = "→" if l.src == obj.id else "←"
                    label = (f"{arrow} {TYPE_ICON.get(oname, '·')} "
                             f"{other.name[:40]}")
                    if st.button(label, key=f"pv_{obj.id}_{lt}_{other_id}",
                                 use_container_width=True):
                        st.session_state["explorer_id"] = other_id
                        st.rerun()
                if len(ls) > 15:
                    st.caption(f"…외 {len(ls) - 15}건")
