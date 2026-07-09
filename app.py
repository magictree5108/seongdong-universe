"""NEO 성동 — 성동구 공공데이터 3D 의미 연결망.

디딤(Didim) 프로젝트가 수집한 성동구 고시공고·자치법규·사전컨설팅 선례를
우주의 성단처럼 시각화한다. 검색하면 관련 자료가 앞으로 끌려나오며
"상황판" 패널에 표시된다.

데이터는 data_pipeline/build.py 가 미리 계산해 data/ 에 커밋해둔 정적
파일(json/npy)이므로, 이 앱은 디딤 백엔드나 원본 데이터 없이 완전히
독립적으로 동작한다.
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "data_pipeline"))
import embedder  # noqa: E402  (경로 삽입 후 임포트)

DATA_DIR = Path(__file__).resolve().parent / "data"
SPREAD = 9.0  # data_pipeline/build.py 의 pca_3d(spread=) 와 맞춘 값

CATEGORY_STYLE = {
    "consulting": {"label": "사전컨설팅·면책 선례", "color": "#ff9d45", "short": "선례"},
    "notice":     {"label": "성동구 고시공고",       "color": "#4fd8ff", "short": "공고"},
    "ordinance":  {"label": "성동구 자치법규",       "color": "#b18bff", "short": "조례"},
}
HIGHLIGHT_COLOR = "#fff3b0"
TOP_K = 14
SIM_DISPLAY_FLOOR = 0.03   # 이보다 낮은 유사도는 상황판에 아예 띄우지 않는다

st.set_page_config(page_title="NEO 성동", page_icon="🛰️",
                    layout="wide", initial_sidebar_state="expanded")


# ── 데이터 로드 ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    nodes = json.loads((DATA_DIR / "nodes.json").read_text(encoding="utf-8"))
    edges = json.loads((DATA_DIR / "edges.json").read_text(encoding="utf-8"))
    meta = json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    vectors = np.load(DATA_DIR / "embeddings.npy")
    idf = np.load(DATA_DIR / "idf.npy")
    return nodes, edges, meta, vectors, idf


# ── 스타일 ───────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }

    .stApp {
        background:
            radial-gradient(1px 1px at 12% 22%, rgba(255,255,255,0.55) 0, transparent 60%),
            radial-gradient(1px 1px at 78% 8%, rgba(255,255,255,0.4) 0, transparent 60%),
            radial-gradient(1.5px 1.5px at 42% 65%, rgba(180,200,255,0.5) 0, transparent 60%),
            radial-gradient(1px 1px at 90% 45%, rgba(255,255,255,0.35) 0, transparent 60%),
            radial-gradient(1px 1px at 25% 85%, rgba(255,255,255,0.4) 0, transparent 60%),
            radial-gradient(1.5px 1.5px at 60% 30%, rgba(180,200,255,0.4) 0, transparent 60%),
            radial-gradient(1px 1px at 8% 55%, rgba(255,255,255,0.3) 0, transparent 60%),
            radial-gradient(1px 1px at 95% 80%, rgba(255,255,255,0.3) 0, transparent 60%),
            linear-gradient(180deg, #05060d 0%, #070a16 55%, #05060d 100%);
        color: #d7e3ff;
    }
    section[data-testid="stSidebar"] {
        background: rgba(6, 9, 18, 0.9);
        border-right: 1px solid rgba(120,150,255,0.18);
    }
    header[data-testid="stHeader"] { background: transparent; }
    div[data-testid="stToolbar"] { visibility: hidden; }

    h1, h2, h3 { color: #eaf1ff !important; letter-spacing: 0.04em; }
    .neo-title {
        font-size: 2.1rem; font-weight: 700; letter-spacing: 0.18em;
        color: #eaf1ff; text-shadow: 0 0 18px rgba(120,170,255,0.55);
        margin-bottom: 0;
    }
    .neo-subtitle { color: #7d8bb0; font-size: 0.85rem; letter-spacing: 0.08em; margin-top: 2px; }

    div[data-testid="stTextInput"] input {
        background: rgba(10,14,26,0.85) !important;
        border: 1px solid rgba(120,170,255,0.4) !important;
        color: #eaf1ff !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.03em;
        border-radius: 4px !important;
        box-shadow: 0 0 14px rgba(80,130,255,0.12) inset;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #4fd8ff !important;
        box-shadow: 0 0 16px rgba(79,216,255,0.45) !important;
    }
    div[data-testid="stTextInput"] label { color: #7d8bb0 !important; }

    .board-card {
        border: 1px solid rgba(120,150,255,0.28);
        background: linear-gradient(180deg, rgba(16,20,36,0.75), rgba(10,13,24,0.75));
        border-radius: 6px; padding: 10px 12px; margin-bottom: 10px;
        box-shadow: 0 0 10px rgba(60,90,200,0.08);
    }
    .board-card .rank { color: #4fd8ff; font-weight: 700; font-size: 0.78rem; }
    .board-card .cat-pill {
        display: inline-block; font-size: 0.68rem; padding: 1px 7px; border-radius: 10px;
        margin-left: 6px; border: 1px solid currentColor;
    }
    .board-card .title { color: #eaf1ff; font-weight: 700; font-size: 0.92rem; margin: 4px 0 2px; }
    .board-card .meta { color: #8291b8; font-size: 0.72rem; margin-bottom: 5px; }
    .board-card .snippet { color: #b9c4e0; font-size: 0.78rem; line-height: 1.45; }
    .board-card .simbar-track { background: rgba(255,255,255,0.08); border-radius: 3px; height: 4px; margin-top: 7px; }
    .board-card .simbar-fill { background: linear-gradient(90deg, #4fd8ff, #fff3b0); height: 4px; border-radius: 3px; }
    .board-card a { color: #4fd8ff; text-decoration: none; font-size: 0.72rem; }
    .board-card a:hover { text-decoration: underline; }

    .idle-box {
        border: 1px dashed rgba(120,150,255,0.3); border-radius: 6px;
        padding: 18px 14px; color: #7d8bb0; font-size: 0.82rem; line-height: 1.6;
    }
    .stat-line { color: #b9c4e0; font-size: 0.82rem; margin: 2px 0; }
    .legend-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
    </style>
    """, unsafe_allow_html=True)


# ── 검색 ─────────────────────────────────────────────────────────

def search(query: str, vectors: np.ndarray, idf: np.ndarray,
          visible_mask: np.ndarray) -> list[tuple[int, float]]:
    """질의와 각 노드의 코사인 유사도 상위 TOP_K를 (노드 인덱스, 점수)로 반환."""
    qvec = embedder.embed([query], idf)[0]
    sims = vectors @ qvec
    sims = np.where(visible_mask, sims, -1.0)
    order = np.argsort(-sims)[:TOP_K]
    return [(int(i), float(sims[i])) for i in order if sims[i] >= SIM_DISPLAY_FLOOR]


# ── 3D 도형 ──────────────────────────────────────────────────────

def build_figure(nodes, edges, visible_mask: np.ndarray,
                 highlight: dict[int, float], query_active: bool):
    fig = go.Figure()

    # 엣지 — 두 끝점이 모두 보이는 경우만
    ex, ey, ez = [], [], []
    for i, j, _w in edges:
        if not (visible_mask[i] and visible_mask[j]):
            continue
        a, b = nodes[i], nodes[j]
        ex += [a["x"], b["x"], None]
        ey += [a["y"], b["y"], None]
        ez += [a["z"], b["z"], None]
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="rgba(120,150,255,0.16)", width=1),
        hoverinfo="skip", showlegend=False,
    ))

    # 카테고리별 기본 노드 (검색 중엔 흐리게)
    for slug, style in CATEGORY_STYLE.items():
        idx = [i for i, n in enumerate(nodes)
               if n["category"] == slug and visible_mask[i] and i not in highlight]
        if not idx:
            continue
        fig.add_trace(go.Scatter3d(
            x=[nodes[i]["x"] for i in idx],
            y=[nodes[i]["y"] for i in idx],
            z=[nodes[i]["z"] for i in idx],
            mode="markers",
            name=style["label"],
            marker=dict(
                size=[4 + min(nodes[i]["degree"], 20) * 0.35 for i in idx],
                color=style["color"],
                opacity=0.18 if query_active else 0.72,
                line=dict(width=0),
            ),
            text=[f'{nodes[i]["title"]}<br>{nodes[i]["source_label"]}<br>{nodes[i]["snippet"]}'
                  for i in idx],
            hovertemplate="%{text}<extra></extra>",
        ))

    # 검색으로 앞으로 끌려나온 노드 + 상황판 빔
    if highlight:
        eye = np.array([1.25, 1.25, 1.25])
        eye = eye / np.linalg.norm(eye)
        front = eye * SPREAD * 1.5

        hx, hy, hz, hcolor, hsize, htext, hline = [], [], [], [], [], [], []
        bx, by, bz = [], [], []
        ranked = sorted(highlight.items(), key=lambda kv: -kv[1])
        for rank, (i, score) in enumerate(ranked):
            n = nodes[i]
            pull = max(0.15, 0.62 - rank * 0.035)
            px = n["x"] * (1 - pull) + front[0] * pull
            py = n["y"] * (1 - pull) + front[1] * pull
            pz = n["z"] * (1 - pull) + front[2] * pull
            hx.append(px); hy.append(py); hz.append(pz)
            hsize.append(max(9, 16 - rank * 0.4))
            hline.append(CATEGORY_STYLE[n["category"]]["color"])
            htext.append(f'{n["title"]}<br>유사도 {score:.2f}')
            bx += [0, px, None]; by += [0, py, None]; bz += [0, pz, None]

        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz, mode="lines",
            line=dict(color="rgba(255,243,176,0.5)", width=2),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=hx, y=hy, z=hz, mode="markers+text",
            marker=dict(size=hsize, color=HIGHLIGHT_COLOR,
                       line=dict(width=2, color=hline), opacity=1.0),
            text=[nodes[i]["title"][:16] + ("…" if len(nodes[i]["title"]) > 16 else "")
                  for i, _s in ranked],
            textposition="top center",
            textfont=dict(color="#fff3b0", size=10),
            hovertext=htext, hovertemplate="%{hovertext}<extra></extra>",
            name="검색 결과", showlegend=False,
        ))

        centroid = np.array([hx, hy, hz]).mean(axis=1) if hx else front
        cam_eye = dict(x=float(centroid[0]) * 0.14 + 1.1,
                      y=float(centroid[1]) * 0.14 + 1.1,
                      z=float(centroid[2]) * 0.14 + 1.1)
    else:
        cam_eye = dict(x=1.25, y=1.25, z=1.25)

    axis = dict(visible=False, showbackground=False, showgrid=False, zeroline=False)
    fig.update_layout(
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis,
                  bgcolor="rgba(0,0,0,0)",
                  camera=dict(eye=cam_eye)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(font=dict(color="#d7e3ff", family="JetBrains Mono"),
                   bgcolor="rgba(6,9,18,0.55)", bordercolor="rgba(120,150,255,0.25)",
                   borderwidth=1, x=0.01, y=0.99),
        uirevision="query" if not query_active else f"query::{query_active}",
        height=720,
    )
    return fig


# ── 상황판 카드 ──────────────────────────────────────────────────

def render_board(nodes, ranked: list[tuple[int, float]]):
    if not ranked:
        st.markdown(
            '<div class="idle-box">STANDBY — 위 검색창에 사업명이나 키워드를 입력하면 '
            '관련 성동구 공공데이터가 앞으로 끌려나오며 여기에 표시됩니다.<br><br>'
            '예: 전동킥보드 주차구역 · 옥상 개방 · 민간위탁 · 청년 마음건강 · 재난지원금</div>',
            unsafe_allow_html=True)
        return
    for rank, (i, score) in enumerate(ranked, start=1):
        n = nodes[i]
        style = CATEGORY_STYLE[n["category"]]
        link = (f'<a href="{n["url"]}" target="_blank" rel="noreferrer">원문 보기 ↗</a>'
                if n["url"] else '<span style="color:#4a5578;">원문 링크 없음</span>')
        st.markdown(f"""
        <div class="board-card">
          <span class="rank">#{rank:02d}</span>
          <span class="cat-pill" style="color:{style['color']};">{style['short']}</span>
          <div class="title">{n['title']}</div>
          <div class="meta">{n['source_label']} · {n['date'] or '날짜 미상'}</div>
          <div class="snippet">{n['snippet']}</div>
          <div class="simbar-track"><div class="simbar-fill" style="width:{max(score,0)*100:.0f}%;"></div></div>
          <div style="margin-top:6px;">{link}</div>
        </div>
        """, unsafe_allow_html=True)


# ── 메인 ─────────────────────────────────────────────────────────

def main():
    inject_css()
    nodes, edges, meta, vectors, idf = load_data()

    st.markdown('<div class="neo-title">NEO 성동</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="neo-subtitle">성동구 공공데이터 3D 의미 연결망 · '
        f'디딤(Didim) 색인 기반 · 노드 {meta["total_nodes"]:,}개 · '
        f'연결 {meta["total_edges"]:,}개</div><br>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 필터")
        active_cats = set()
        for slug, style in CATEGORY_STYLE.items():
            n_count = meta["categories"].get(slug, 0)
            checked = st.checkbox(
                f'{style["label"]} ({n_count:,})', value=True, key=f"cat_{slug}")
            st.markdown(
                f'<span class="legend-dot" style="background:{style["color"]};"></span>',
                unsafe_allow_html=True)
            if checked:
                active_cats.add(slug)

        st.markdown("---")
        st.markdown("### 정보")
        st.markdown(f'<div class="stat-line">데이터 빌드: {meta["built_at"]}</div>',
                   unsafe_allow_html=True)
        st.markdown(
            '<div class="stat-line" style="margin-top:10px; color:#5c6890;">'
            '공개 행정데이터 기반 데모입니다. 원문은 각 카드의 링크에서 확인하세요. '
            '적법성·정확성을 보증하지 않습니다.</div>', unsafe_allow_html=True)

    visible_mask = np.array([n["category"] in active_cats for n in nodes])

    col_viz, col_board = st.columns([2.5, 1], gap="medium")

    with col_viz:
        query = st.text_input(
            "검색", placeholder="사업명·키워드를 입력하고 Enter — 예: 전동킥보드 주차구역",
            label_visibility="collapsed")
        highlight: dict[int, float] = {}
        ranked: list[tuple[int, float]] = []
        if query.strip():
            ranked = search(query.strip(), vectors, idf, visible_mask)
            highlight = dict(ranked)
        fig = build_figure(nodes, edges, visible_mask, highlight,
                          query_active=query.strip())
        st.plotly_chart(fig, width="stretch",
                       config={"displaylogo": False})

    with col_board:
        st.markdown("#### 상황판")
        render_board(nodes, ranked)


if __name__ == "__main__":
    main()
