"""NEO 성동 — 성동구 공공데이터 3D 의미 연결망 (팔란티어 Gotham 오마주).

디딤(Didim) 프로젝트가 수집한 성동구 고시공고·자치법규·사전컨설팅 선례를
우주의 성단처럼 시각화하고, 그 위에 온톨로지 레이어(부서·법령·동네 개체)를
얹는다. 검색하면 관련 자료가 앞으로 끌려나오고(상황판), 노드를 선택하면
속성과 연결을 보여주는 Object 360 패널에서 연결을 타고 피벗할 수 있으며,
조사 중 발견한 자료는 케이스 파일에 핀으로 모아 리포트로 내보낸다.

데이터는 data_pipeline/build.py 가 미리 계산해 data/ 에 커밋해둔 정적
파일(json/npy)이므로, 이 앱은 디딤 백엔드 없이 완전히 독립적으로 동작한다.
"""
import html
import json
from collections import defaultdict
from datetime import datetime
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
ENTITY_STYLE = {
    "dept":  {"label": "담당 부서",        "color": "#6ee7a8", "short": "부서"},
    "law":   {"label": "법령·자치법규 참조", "color": "#ff7b9c", "short": "법령"},
    "place": {"label": "성동구 동네",       "color": "#ffd166", "short": "동네"},
}
EDGE_COLOR = {
    "sim":   "rgba(120,150,255,0.13)",
    "dept":  "rgba(110,231,168,0.20)",
    "law":   "rgba(255,123,156,0.16)",
    "place": "rgba(255,209,102,0.20)",
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


@st.cache_data(show_spinner=False)
def build_adjacency():
    """피벗용 인접 구조 — 유사도 이웃, 문서→개체, (개체→문서는 node.links)."""
    nodes, edges, _meta, _v, _i = load_data()
    sim_nbrs: dict[int, list] = defaultdict(list)
    for i, j, w, t in edges:
        if t == "sim":
            sim_nbrs[i].append((j, w))
            sim_nbrs[j].append((i, w))
    doc_entities: dict[int, list] = defaultdict(list)
    for n in nodes:
        if n["kind"] == "entity":
            for d in n["links"]:
                doc_entities[d].append(n["id"])
    return dict(sim_nbrs), dict(doc_entities)


def node_style(n) -> dict:
    return ENTITY_STYLE[n["etype"]] if n["kind"] == "entity" else CATEGORY_STYLE[n["category"]]


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
            linear-gradient(180deg, #05060d 0%, #070a16 55%, #05060d 100%);
        color: #d7e3ff;
    }
    section[data-testid="stSidebar"] {
        background: rgba(6, 9, 18, 0.9);
        border-right: 1px solid rgba(120,150,255,0.18);
    }
    header[data-testid="stHeader"] { background: transparent; }
    div[data-testid="stToolbar"] { visibility: hidden; }

    h1, h2, h3, h4 { color: #eaf1ff !important; letter-spacing: 0.04em; }

    .classification {
        text-align: center; font-size: 0.68rem; letter-spacing: 0.35em;
        color: #6ee7a8; border: 1px solid rgba(110,231,168,0.35);
        background: rgba(110,231,168,0.06);
        padding: 3px 0; margin-bottom: 14px; border-radius: 3px;
    }
    .neo-title {
        font-size: 2.1rem; font-weight: 700; letter-spacing: 0.18em;
        color: #eaf1ff; text-shadow: 0 0 18px rgba(120,170,255,0.55);
        margin-bottom: 0;
    }
    .neo-subtitle { color: #7d8bb0; font-size: 0.82rem; letter-spacing: 0.08em; margin-top: 2px; }

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

    div.stButton > button {
        background: rgba(12,17,32,0.9); color: #9fb4e8;
        border: 1px solid rgba(120,150,255,0.3); border-radius: 3px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
        padding: 2px 10px; min-height: 26px; letter-spacing: 0.05em;
    }
    div.stButton > button:hover {
        border-color: #4fd8ff; color: #4fd8ff; background: rgba(79,216,255,0.08);
    }
    div[data-testid="stDownloadButton"] > button {
        background: rgba(110,231,168,0.08); color: #6ee7a8;
        border: 1px solid rgba(110,231,168,0.4); border-radius: 3px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    }

    .board-card, .obj-card {
        border: 1px solid rgba(120,150,255,0.28);
        background: linear-gradient(180deg, rgba(16,20,36,0.75), rgba(10,13,24,0.75));
        border-radius: 6px; padding: 10px 12px; margin-bottom: 4px;
        box-shadow: 0 0 10px rgba(60,90,200,0.08);
    }
    .board-card .rank { color: #4fd8ff; font-weight: 700; font-size: 0.78rem; }
    .cat-pill {
        display: inline-block; font-size: 0.68rem; padding: 1px 7px; border-radius: 10px;
        margin-left: 6px; border: 1px solid currentColor;
    }
    .board-card .title, .obj-card .title { color: #eaf1ff; font-weight: 700; font-size: 0.92rem; margin: 4px 0 2px; }
    .board-card .meta, .obj-card .meta { color: #8291b8; font-size: 0.72rem; margin-bottom: 5px; }
    .board-card .snippet, .obj-card .snippet { color: #b9c4e0; font-size: 0.78rem; line-height: 1.45; }
    .simbar-track { background: rgba(255,255,255,0.08); border-radius: 3px; height: 4px; margin-top: 7px; }
    .simbar-fill { background: linear-gradient(90deg, #4fd8ff, #fff3b0); height: 4px; border-radius: 3px; }
    .board-card a, .obj-card a { color: #4fd8ff; text-decoration: none; font-size: 0.72rem; }
    .board-card a:hover, .obj-card a:hover { text-decoration: underline; }

    .obj-header { color: #4fd8ff; font-size: 0.72rem; letter-spacing: 0.25em; margin: 10px 0 4px; }
    .prop-table { width: 100%; font-size: 0.72rem; color: #b9c4e0; border-collapse: collapse; }
    .prop-table td { border-top: 1px solid rgba(120,150,255,0.12); padding: 3px 4px; }
    .prop-table td:first-child { color: #7d8bb0; width: 34%; }

    .idle-box {
        border: 1px dashed rgba(120,150,255,0.3); border-radius: 6px;
        padding: 18px 14px; color: #7d8bb0; font-size: 0.82rem; line-height: 1.6;
    }
    .status-bar {
        border-top: 1px solid rgba(120,150,255,0.25);
        color: #7d8bb0; font-size: 0.7rem; letter-spacing: 0.06em;
        padding: 6px 2px 0; margin-top: 10px; white-space: nowrap;
        overflow-x: auto;
    }
    .status-bar b { color: #6ee7a8; font-weight: 500; }
    .stat-line { color: #b9c4e0; font-size: 0.82rem; margin: 2px 0; }
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

def _neighbor_ids(node_id: int, nodes) -> list[tuple[int, float]]:
    """선택 노드의 연결(유사 문서 + 개체 링크)을 (id, 강도)로 반환."""
    sim_nbrs, doc_entities = build_adjacency()
    n = nodes[node_id]
    out: dict[int, float] = {}
    if n["kind"] == "entity":
        for d in n["links"]:
            out[d] = max(out.get(d, 0), 1.0)
    else:
        for j, w in sorted(sim_nbrs.get(node_id, []), key=lambda x: -x[1])[:10]:
            out[j] = max(out.get(j, 0), w)
        for e in doc_entities.get(node_id, []):
            out[e] = max(out.get(e, 0), 1.0)
    out.pop(node_id, None)
    return sorted(out.items(), key=lambda x: -x[1])


def build_figure(nodes, edges, visible_mask, search_ranked, sel_id):
    fig = go.Figure()
    sel_mode = sel_id is not None and visible_mask[sel_id]
    query_mode = bool(search_ranked) and not sel_mode
    highlight = dict(search_ranked) if query_mode else {}
    focus_ids = set(highlight)
    if sel_mode:
        nbrs = _neighbor_ids(sel_id, nodes)
        focus_ids = {sel_id} | {i for i, _w in nbrs}

    # 타입별 엣지 (양 끝이 보일 때만). 포커스 모드에선 배경 연결을 아예
    # 그리지 않는다 — 반투명 선 수천 개가 겹치면 하얗게 포화되기 때문.
    if not (sel_mode or query_mode):
        by_type: dict[str, list] = defaultdict(lambda: ([], [], []))
        for i, j, _w, t in edges:
            if not (visible_mask[i] and visible_mask[j]):
                continue
            xs, ys, zs = by_type[t]
            a, b = nodes[i], nodes[j]
            xs += [a["x"], b["x"], None]
            ys += [a["y"], b["y"], None]
            zs += [a["z"], b["z"], None]
        for t, (xs, ys, zs) in by_type.items():
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=EDGE_COLOR[t], width=1),
                hoverinfo="skip", showlegend=False))

    # 기본 노드: 문서(원) — 카테고리별 / 개체(다이아몬드) — 타입별
    groups = [(f'doc:{slug}', style, lambda n, s=slug: n["kind"] == "doc" and n["category"] == s, "circle")
              for slug, style in CATEGORY_STYLE.items()]
    groups += [(f'ent:{et}', style, lambda n, e=et: n["kind"] == "entity" and n["etype"] == e, "diamond")
               for et, style in ENTITY_STYLE.items()]
    dim = sel_mode or query_mode
    for gid, style, pred, symbol in groups:
        idx = [i for i, n in enumerate(nodes)
               if pred(n) and visible_mask[i] and i not in focus_ids]
        if not idx:
            continue
        is_entity = symbol == "diamond"
        fig.add_trace(go.Scatter3d(
            x=[nodes[i]["x"] for i in idx],
            y=[nodes[i]["y"] for i in idx],
            z=[nodes[i]["z"] for i in idx],
            mode="markers",
            name=style["label"],
            marker=dict(
                size=[(7 + min(nodes[i]["degree"], 160) * 0.045) if is_entity
                      else (4 + min(nodes[i]["degree"], 20) * 0.35) for i in idx],
                color=style["color"], symbol=symbol,
                opacity=0.15 if dim else (0.95 if is_entity else 0.72),
                line=dict(width=1, color=style["color"]) if is_entity else dict(width=0),
            ),
            text=[f'{nodes[i]["title"]}<br>{nodes[i]["source_label"]}' for i in idx],
            hovertemplate="%{text}<extra></extra>",
        ))

    cam_eye = dict(x=1.25, y=1.25, z=1.25)

    # 검색 모드 — 결과를 카메라 앞으로 끌어오고 원점 빔을 쏜다 (상황판 연출)
    if query_mode:
        eye = np.array([1.25, 1.25, 1.25]); eye = eye / np.linalg.norm(eye)
        front = eye * SPREAD * 1.5
        hx, hy, hz, hsize, hline, htext, labels = [], [], [], [], [], [], []
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
            hline.append(node_style(n)["color"])
            htext.append(f'{n["title"]}<br>유사도 {score:.2f}')
            labels.append(n["title"][:16] + ("…" if len(n["title"]) > 16 else "")
                          if rank < 8 else "")
            bx += [0, px, None]; by += [0, py, None]; bz += [0, pz, None]
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz, mode="lines",
            line=dict(color="rgba(255,243,176,0.5)", width=2),
            hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter3d(
            x=hx, y=hy, z=hz, mode="markers+text",
            marker=dict(size=hsize, color=HIGHLIGHT_COLOR,
                       line=dict(width=2, color=hline), opacity=1.0),
            text=labels, textposition="top center",
            textfont=dict(color="#fff3b0", size=10),
            hovertext=htext, hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False))
        centroid = np.array([hx, hy, hz]).mean(axis=1)
        cam_eye = dict(x=float(centroid[0]) * 0.14 + 1.1,
                      y=float(centroid[1]) * 0.14 + 1.1,
                      z=float(centroid[2]) * 0.14 + 1.1)

    # 선택 모드 — 성좌(constellation): 제자리에서 선택 노드와 연결만 밝힌다
    if sel_mode:
        s = nodes[sel_id]
        nbrs = [(i, w) for i, w in _neighbor_ids(sel_id, nodes) if visible_mask[i]]
        bx, by, bz = [], [], []
        for i, _w in nbrs:
            n = nodes[i]
            bx += [s["x"], n["x"], None]
            by += [s["y"], n["y"], None]
            bz += [s["z"], n["z"], None]
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz, mode="lines",
            line=dict(color="rgba(255,243,176,0.65)", width=3),
            hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter3d(
            x=[nodes[i]["x"] for i, _w in nbrs],
            y=[nodes[i]["y"] for i, _w in nbrs],
            z=[nodes[i]["z"] for i, _w in nbrs],
            mode="markers",
            marker=dict(
                size=[9 if nodes[i]["kind"] == "entity" else 7 for i, _w in nbrs],
                color=[node_style(nodes[i])["color"] for i, _w in nbrs],
                symbol=["diamond" if nodes[i]["kind"] == "entity" else "circle"
                        for i, _w in nbrs],
                opacity=1.0, line=dict(width=1, color="#eaf1ff")),
            text=[nodes[i]["title"] for i, _w in nbrs],
            hovertemplate="%{text}<extra></extra>", showlegend=False))
        fig.add_trace(go.Scatter3d(
            x=[s["x"]], y=[s["y"]], z=[s["z"]], mode="markers+text",
            marker=dict(size=18, color=HIGHLIGHT_COLOR,
                       symbol="diamond" if s["kind"] == "entity" else "circle",
                       line=dict(width=3, color="#ffffff"), opacity=1.0),
            text=[s["title"][:20]], textposition="top center",
            textfont=dict(color="#ffffff", size=11),
            hovertext=[s["title"]], hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False))
        cam_eye = dict(x=s["x"] * 0.05 + 1.2, y=s["y"] * 0.05 + 1.2,
                      z=s["z"] * 0.05 + 1.2)

    axis = dict(visible=False, showbackground=False, showgrid=False, zeroline=False)
    fig.update_layout(
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis,
                  bgcolor="rgba(0,0,0,0)", camera=dict(eye=cam_eye)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(font=dict(color="#d7e3ff", family="JetBrains Mono", size=10),
                   bgcolor="rgba(6,9,18,0.55)", bordercolor="rgba(120,150,255,0.25)",
                   borderwidth=1, x=0.01, y=0.99),
        uirevision=f"focus::{sel_id}::{bool(search_ranked)}",
        height=640,
    )
    return fig


# ── 패널 구성요소 ────────────────────────────────────────────────

def _safe_url(url: str | None) -> str | None:
    """href 주입 방지 — http(s) 스킴만 허용한다 (javascript: 등 차단)."""
    if url and url.startswith(("http://", "https://")):
        return url
    return None


def _select_node(i: int | None):
    st.session_state.sel = i


def _pin_node(i: int):
    if i not in st.session_state.case:
        st.session_state.case.append(i)


def _unpin_node(i: int):
    st.session_state.case = [c for c in st.session_state.case if c != i]


def _card_html(n, rank: int | None = None, score: float | None = None) -> str:
    """검색/케이스 카드 HTML. 크롤링 원문은 반드시 이스케이프한다."""
    style = node_style(n)
    title = html.escape(n["title"])
    source_label = html.escape(n["source_label"])
    date = html.escape(n["date"]) if n["date"] else "날짜 미상"
    snippet = html.escape(n["snippet"])
    safe_url = _safe_url(n["url"])
    link = (f'<a href="{html.escape(safe_url)}" target="_blank" rel="noopener noreferrer">원문 보기 ↗</a>'
            if safe_url else '<span style="color:#4a5578;">원문 링크 없음</span>')
    rank_html = f'<span class="rank">#{rank:02d}</span>' if rank else ""
    simbar = (f'<div class="simbar-track"><div class="simbar-fill" '
              f'style="width:{max(score, 0) * 100:.0f}%;"></div></div>'
              if score is not None else "")
    return f"""
    <div class="board-card">
      {rank_html}
      <span class="cat-pill" style="color:{style['color']};">{style['short']}</span>
      <div class="title">{title}</div>
      <div class="meta">{source_label} · {date}</div>
      <div class="snippet">{snippet}</div>
      {simbar}
      <div style="margin-top:6px;">{link}</div>
    </div>"""


def _card_buttons(i: int, key_prefix: str):
    c1, c2, _sp = st.columns([1.1, 1, 1.8])
    with c1:
        st.button("OBJECT 360°", key=f"{key_prefix}_sel_{i}",
                  on_click=_select_node, args=(i,))
    with c2:
        pinned = i in st.session_state.case
        st.button("PINNED ✓" if pinned else "+ CASE", key=f"{key_prefix}_pin_{i}",
                  on_click=_pin_node, args=(i,), disabled=pinned)


def render_board(nodes, ranked):
    if not ranked:
        st.markdown(
            '<div class="idle-box">STANDBY — 검색창에 사업명이나 키워드를 입력하면 '
            '관련 데이터가 앞으로 끌려나오며 여기에 표시됩니다. 카드의 OBJECT 360° 로 '
            '연결을 타고 이동하고, + CASE 로 케이스 파일에 모으세요.<br><br>'
            '예: 전동킥보드 주차구역 · 민간위탁 · 청년 마음건강 · 재난지원금 · 성수동</div>',
            unsafe_allow_html=True)
        return
    for rank, (i, score) in enumerate(ranked, start=1):
        st.markdown(_card_html(nodes[i], rank=rank, score=score),
                   unsafe_allow_html=True)
        _card_buttons(i, "board")


def _prop_rows(n) -> str:
    sim_nbrs, doc_entities = build_adjacency()
    if n["kind"] == "entity":
        links = f'문서 {n["degree"]}건'
    else:
        links = (f'유사 문서 {len(sim_nbrs.get(n["id"], []))} · '
                 f'개체 {len(doc_entities.get(n["id"], []))}')
    rows = [
        ("OBJECT ID", n["doc_id"]),
        ("유형", f'{node_style(n)["label"]} ({ "개체" if n["kind"] == "entity" else "문서"})'),
        ("날짜", n["date"] or "—"),
        ("연결", links),
    ]
    return "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>"
                   for k, v in rows)


def render_object_360(nodes, sel_id: int):
    n = nodes[sel_id]
    st.button("← 상황판으로", key="close360", on_click=_select_node, args=(None,))
    st.markdown(_card_html(n), unsafe_allow_html=True)
    _card_buttons(sel_id, "obj")
    st.markdown('<div class="obj-header">PROPERTIES</div>', unsafe_allow_html=True)
    st.markdown(f'<table class="prop-table">{_prop_rows(n)}</table>',
               unsafe_allow_html=True)

    sim_nbrs, doc_entities = build_adjacency()
    if n["kind"] == "doc":
        ents = doc_entities.get(sel_id, [])
        if ents:
            st.markdown('<div class="obj-header">LINKED OBJECTS — 개체</div>',
                       unsafe_allow_html=True)
            for e in ents:
                en = nodes[e]
                st.button(f'◇ {en["title"]}  [{node_style(en)["short"]} · {en["degree"]}건]',
                          key=f"piv_ent_{e}", on_click=_select_node, args=(e,))
        near = sorted(sim_nbrs.get(sel_id, []), key=lambda x: -x[1])[:6]
        if near:
            st.markdown('<div class="obj-header">LINKED OBJECTS — 유사 문서</div>',
                       unsafe_allow_html=True)
            for j, w in near:
                jn = nodes[j]
                title = jn["title"][:30] + ("…" if len(jn["title"]) > 30 else "")
                st.button(f'○ {title}  [{node_style(jn)["short"]} · {w:.2f}]',
                          key=f"piv_doc_{j}", on_click=_select_node, args=(j,))
    else:
        linked = n["links"]
        linked_nodes = sorted((nodes[d] for d in linked),
                              key=lambda x: (x["date"] or ""), reverse=True)
        st.markdown(
            f'<div class="obj-header">LINKED OBJECTS — 연결 문서 {len(linked)}건 (최신순)</div>',
            unsafe_allow_html=True)
        for jn in linked_nodes[:12]:
            title = jn["title"][:30] + ("…" if len(jn["title"]) > 30 else "")
            st.button(f'○ {title}  [{jn["date"] or "날짜 미상"}]',
                      key=f"piv_link_{jn['id']}", on_click=_select_node, args=(jn["id"],))
        if len(linked) > 12:
            st.markdown(f'<div class="stat-line">… 외 {len(linked) - 12}건</div>',
                       unsafe_allow_html=True)


def _case_markdown(nodes, case: list[int]) -> str:
    lines = ["# NEO 성동 — 케이스 파일",
             f"- 내보낸 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             f"- 수집 항목: {len(case)}건",
             "",
             "> 공개 행정데이터 기반 데모에서 수집한 자료 목록입니다. "
             "내용의 정확성·적법성은 원문에서 확인하십시오.", ""]
    for k, i in enumerate(case, start=1):
        n = nodes[i]
        lines += [f"## {k}. {n['title']}",
                  f"- 출처: {n['source_label']} · {n['date'] or '날짜 미상'}",
                  f"- 원문: {n['url'] or '링크 없음'}",
                  f"- 발췌: {n['snippet']}", ""]
    return "\n".join(lines)


def render_case_file(nodes):
    case = st.session_state.case
    with st.expander(f"▣ 케이스 파일 ({len(case)}건)", expanded=bool(case)):
        if not case:
            st.markdown('<div class="stat-line">카드의 + CASE 버튼으로 자료를 모으세요.</div>',
                       unsafe_allow_html=True)
            return
        for i in case:
            n = nodes[i]
            title = n["title"][:34] + ("…" if len(n["title"]) > 34 else "")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(
                    f'<div class="stat-line" style="color:{node_style(n)["color"]};">'
                    f'▪ {html.escape(title)}</div>', unsafe_allow_html=True)
            with c2:
                st.button("제거", key=f"unpin_{i}", on_click=_unpin_node, args=(i,))
        st.download_button(
            "케이스 파일 내보내기 (.md)", _case_markdown(nodes, case),
            file_name=f"neo-seongdong-case-{datetime.now().strftime('%Y%m%d-%H%M')}.md",
            mime="text/markdown")


# ── 메인 ─────────────────────────────────────────────────────────

def main():
    inject_css()
    nodes, edges, meta, vectors, idf = load_data()
    st.session_state.setdefault("sel", None)
    st.session_state.setdefault("case", [])

    st.markdown(
        '<div class="classification">OPEN DATA ── 성동구 공개 행정데이터 ── 데모 · 판단 없음</div>',
        unsafe_allow_html=True)
    st.markdown('<div class="neo-title">NEO 성동</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="neo-subtitle">성동구 공공데이터 3D 의미 연결망 + 온톨로지 · '
        f'문서 {meta["total_docs"]:,} · 개체 {meta["total_entities"]} · '
        f'연결 {meta["total_edges"]:,}</div><br>', unsafe_allow_html=True)

    # 사이드바 — 레이어 토글 + 기간 필터
    with st.sidebar:
        st.markdown("### 레이어")
        active_cats = {slug for slug, style in CATEGORY_STYLE.items()
                       if st.checkbox(f'● {style["label"]} ({meta["categories"].get(slug, 0):,})',
                                      value=True, key=f"cat_{slug}")}
        st.markdown("### 온톨로지")
        active_etypes = {et for et, style in ENTITY_STYLE.items()
                         if st.checkbox(f'◇ {style["label"]} ({meta["entity_types"].get(et, 0)})',
                                        value=True, key=f"ent_{et}")}
        st.markdown("---")
        st.markdown("### 기간")
        years = [n["year"] for n in nodes if n["year"]]
        y_min, y_max = min(years), max(years)
        y_range = st.slider("연도 범위", y_min, y_max, (y_min, y_max),
                            label_visibility="collapsed")
        include_undated = st.checkbox("날짜 미상 포함", value=True)
        st.markdown("---")
        st.markdown(f'<div class="stat-line">데이터 빌드: {meta["built_at"][:16]}</div>',
                   unsafe_allow_html=True)
        st.markdown(
            '<div class="stat-line" style="margin-top:8px; color:#5c6890;">'
            '공개 행정데이터 기반 데모입니다. 개체는 규칙 기반 자동 추출로 '
            '오추출이 있을 수 있습니다. 적법성·정확성을 보증하지 않습니다.</div>',
            unsafe_allow_html=True)

    def _visible(n) -> bool:
        if n["kind"] == "entity":
            return n["etype"] in active_etypes
        if n["category"] not in active_cats:
            return False
        if n["year"] is None:
            return include_undated
        return y_range[0] <= n["year"] <= y_range[1]

    visible_mask = np.array([_visible(n) for n in nodes])

    col_viz, col_panel = st.columns([2.5, 1], gap="medium")

    with col_viz:
        query = st.text_input(
            "검색", placeholder="QUERY ▸ 사업명·키워드 입력 후 Enter — 예: 전동킥보드 주차구역",
            label_visibility="collapsed")
        ranked = (search(query.strip(), vectors, idf, visible_mask)
                  if query.strip() else [])
        sel_id = st.session_state.sel
        if sel_id is not None and not visible_mask[sel_id]:
            sel_id = None   # 필터로 가려진 선택은 해제된 것으로 취급
        fig = build_figure(nodes, edges, visible_mask, ranked, sel_id)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

        vis_docs = int(sum(1 for i, n in enumerate(nodes)
                           if visible_mask[i] and n["kind"] == "doc"))
        vis_ents = int(sum(1 for i, n in enumerate(nodes)
                           if visible_mask[i] and n["kind"] == "entity"))
        sel_title = (html.escape(nodes[sel_id]["title"][:24]) if sel_id is not None else "—")
        q_text = html.escape(query.strip()[:24]) if query.strip() else "—"
        st.markdown(
            f'<div class="status-bar">SYS <b>▮ ONLINE</b> · 문서 {vis_docs:,}/{meta["total_docs"]:,}'
            f' · 개체 {vis_ents}/{meta["total_entities"]} · 기간 {y_range[0]}–{y_range[1]}'
            f' · QUERY "{q_text}" · SELECT {sel_title}'
            f' · CASE {len(st.session_state.case)}건</div>',
            unsafe_allow_html=True)

    with col_panel:
        if sel_id is not None:
            st.markdown("#### ▣ OBJECT 360")
            render_object_360(nodes, sel_id)
        else:
            st.markdown("#### ▣ 상황판")
            render_board(nodes, ranked)
        render_case_file(nodes)


if __name__ == "__main__":
    main()
