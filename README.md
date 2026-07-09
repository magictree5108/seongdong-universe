# NEO 성동

성동구 공공데이터를 3D 의미 연결망으로 시각화한 Streamlit 앱입니다.
[디딤(Didim)](../didim) 프로젝트가 수집·색인한 성동구 고시공고·자치법규·
사전컨설팅(감사원·자체감사기구) 선례 2,671건을 우주의 성단처럼 배치하고,
검색하면 관련 자료가 앞으로 끌려나오며 "상황판" 패널에 표시됩니다.

- 노드: 문서 한 건 (제목·출처·날짜·발췌·원문 링크)
- 노드 간 선: 의미적으로 가까운 문서끼리의 연결 (코사인 유사도 k-최근접 이웃)
- 검색창: 질의와 각 노드의 유사도를 계산해 상위 결과를 앞으로 끌어오고,
  오른쪽 상황판에 카드로 정리해 보여줍니다

## 실행 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 http://localhost:8501 접속. 데스크톱/와이드 화면을 권장합니다.

데이터(`data/nodes.json`, `edges.json`, `embeddings.npy`, `idf.npy`)는 이미
빌드되어 저장소에 포함돼 있으므로, 위 실행만으로 바로 동작합니다 — 디딤
백엔드나 원본 데이터가 없어도 됩니다.

## 데이터 재빌드

디딤 프로젝트의 로컬 색인(`~/didim/data/index/*.meta.json`)이 갱신되었을 때
NEO 성동 데이터를 다시 생성하려면:

```bash
python data_pipeline/build.py --didim-index-dir ~/didim/data/index
```

임베딩은 문자 n-gram 해싱 TF-IDF(외부 모델·API 불필요, `data_pipeline/embedder.py`)이며,
디딤의 색인 임베더와 같은 방식을 독립적으로 재구현한 것입니다. 3D 좌표는
순수 numpy PCA(SVD)로 계산합니다.

## 구조

```
neo-seongdong/
├── app.py                  # Streamlit 앱 (시각화 + 검색 + 상황판)
├── data_pipeline/
│   ├── embedder.py          # 해싱 n-gram TF-IDF 임베더 (빌드·검색 공용)
│   └── build.py             # 디딤 색인 → nodes/edges/embeddings 빌드 스크립트
├── data/                    # 빌드 산출물 (저장소에 커밋됨, 약 13MB)
│   ├── nodes.json           # 노드 메타데이터 + 3D 좌표
│   ├── edges.json           # [i, j, 유사도] 목록
│   ├── embeddings.npy       # 노드별 1024차원 벡터 (검색용)
│   ├── idf.npy              # 검색 질의 임베딩에 쓰는 IDF 벡터
│   └── meta.json            # 빌드 통계(노드·엣지 수, 빌드 시각 등)
└── .streamlit/config.toml   # 다크 테마
```

## Streamlit Community Cloud 배포

1. 이 저장소를 GitHub에 푸시
2. https://share.streamlit.io 에서 새 앱 생성 → 이 저장소·`app.py` 선택
3. `requirements.txt`가 자동 인식되어 별도 설정 없이 배포됩니다

## 고지

이 프로젝트는 성동구가 공개한 행정데이터(고시공고·자치법규·사전컨설팅 처리사례)를
재구성한 시각화 데모입니다. 검색 결과의 관련성은 문자 기반 통계적 유사도로
계산된 것으로 내용의 정확성·최신성·적법성을 보증하지 않습니다. 정확한 내용은
각 카드의 원문 링크에서 확인하십시오.
