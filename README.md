# 성동 UNIVERSE

성동구 공공데이터를 **상시 유동하는 3D 의미 우주**로 그리는 Streamlit 앱입니다.
팔란티어 Gotham의 조사(investigation) UX에서 영감을 받았습니다.

[디딤(Didim)](../didim) 프로젝트가 수집·색인한 성동구 공공데이터
**5,208건**(고시공고 1,293 · 자치법규 644 · 사전컨설팅/면책 선례 793 ·
조직·업무분장 41 · 보도/새소식/감사결과 2,581)과 규칙 기반 추출 개체
120종(부서·법령·동네)을 성단으로 배치합니다.

## 인터랙션

1. **우주**: 성단이 느리게 자전하고 노드들이 미세하게 부유한다.
   드래그 회전 · 휠 확대/축소.
2. **검색**: 성단 한가운데의 검색창에 입력하면 —
   Claude가 질의를 행정용어로 확장(마음건강→심리상담·정신건강)하고,
   어휘 일치 후보 중 실제 관련 자료만 선별한다 (기본 모델 **sonnet 5**,
   사이드바에서 haiku로 전환 가능).
3. **딥 줌**: 검색이 끝나면 검색창이 사라지고 카메라가 결과 성단 속으로
   줌인. 결과 카드가 우주 위에 반투명(80%)으로 떠서 자기 노드와
   커넥터 라인으로 연결된다 — 커서를 올리면 선명해진다(100%).
4. **국가 법령**: 법제처 국가법령정보센터 실시간 검색으로 상위법을
   함께 표시 (LAW_OC 키 필요).
5. **초기화**: 우상단 ⟲ 버튼으로 전체 우주로 복귀.

## 실행 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 http://localhost:8501 접속. 데스크톱/와이드 화면 권장.
데이터(`data/`)는 빌드되어 저장소에 포함돼 있으므로 바로 동작합니다 —
디딤 백엔드나 원본 데이터가 없어도 됩니다.

### AI 검색 활성화 (권장)

`.streamlit/secrets.toml`(git 제외됨)에 키를 넣으면 AI 검색·국가법령이 켜집니다:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
LAW_OC = "법제처 OC 아이디"   # 선택 — 국가 법령 실시간 검색
```

키가 없으면 원 질의의 어휘 일치 검색으로만 동작합니다(확장·선별 없음).
기본 모델은 `NEO_LLM_MODEL` 환경변수로 바꿀 수 있습니다.

참고: Claude 5 계열(sonnet 5)은 응답 앞에 thinking 블록을 생성하므로
구조화 출력 호출의 max_tokens 에 여유가 필요합니다 (app.py 참조).

## 데이터 재빌드

디딤 프로젝트의 로컬 색인(`~/didim/data/index/*.meta.json`)이 갱신되었을 때:

```bash
python data_pipeline/build.py --didim-index-dir ~/didim/data/index
```

- 임베딩: 문자 n-gram 해싱 TF-IDF (외부 모델·API 불필요)
- 3D 좌표: 순수 numpy PCA(SVD) · 의미 연결망: 코사인 k-NN
- 개체 추출: 정규식·사전 기반 — LLM 없이 결정적으로 동작

## 온톨로지 계층 (팔란티어 의미 계층의 경량 재현)

문서 중심 그래프와 별도로, 부서·사업·조례·예산 등을 1급 객체로 다루는
읽기 전용 의미 계층을 구축 중입니다 — 8개 객체 타입(Policy, Department,
Ordinance, BudgetItem, PressRelease, Facility, District, ComplaintType)과
8개 링크 타입(담당·집행·근거·언급·관할·위치·소관)을 Pydantic 스키마로
버전 관리하고 SQLite + NetworkX에 저장합니다. 팔란티어의 Action(쓰기/실행)
계층은 의도적으로 구현하지 않습니다.

```bash
python -m ontology.promote   # 디딤 색인 → data/ontology.db (조례·부서·보도 승격)
python -m ontology.verify    # 벤치마크: 빈 그래프 왕복·링크 규칙·다중 홉 조회
```

현재 승격된 객체: Ordinance 644 · Department 41 · PressRelease 2,330.
Policy·BudgetItem은 예산 데이터 수집(2단계) 후 생성됩니다.

## 검색 품질 평가

무작위 공무원 질의 100개로 파이프라인을 채점하는 하니스가 있습니다:

```bash
ANTHROPIC_API_KEY=... python eval/run_eval.py           # 기존 질문으로 재평가
ANTHROPIC_API_KEY=... python eval/run_eval.py --regen   # 질문 재생성
```

주의: 심사관 프롬프트를 바꾸면 버전 간 점수 비교가 불가능해집니다 —
비교할 땐 같은 심사관으로 두 결과를 재채점하십시오.

## 구조

```
neo-seongdong/
├── app.py                   # Streamlit 엔트리 (검색 파이프라인 + 컴포넌트 호출)
├── component/
│   ├── index.html           # 성동 UNIVERSE 렌더러 (Three.js, 커스텀 컴포넌트)
│   └── three.min.js         # 번들된 Three.js (r128 UMD)
├── data_pipeline/
│   ├── embedder.py          # 해싱 n-gram TF-IDF 임베더 (빌드·검색 공용)
│   └── build.py             # 디딤 색인 → 노드/엣지/온톨로지/임베딩 빌드
├── ontology/
│   ├── schema.py            # 8개 객체·8개 링크 타입 (Pydantic, 코드로 버전 관리)
│   ├── store.py             # SQLite 저장소 + NetworkX 그래프 질의
│   ├── promote.py           # 디딤 색인 → 온톨로지 객체 승격
│   └── verify.py            # 1단계 벤치마크 (왕복·링크 규칙·다중 홉)
├── data/                    # 빌드 산출물 (저장소에 커밋됨, 약 33MB)
├── eval/                    # 100문항 검색 품질 평가 하니스 + 리포트
└── .streamlit/config.toml   # 다크 테마
```

## Streamlit Community Cloud 배포

1. 이 저장소를 GitHub에 푸시
2. https://share.streamlit.io 에서 새 앱 생성 → 이 저장소·`app.py` 선택
3. Secrets 설정에 `ANTHROPIC_API_KEY`(·선택 `LAW_OC`) 입력

## 고지

이 프로젝트는 성동구가 공개한 행정데이터를 재구성한 시각화 데모입니다.
개체(부서·법령·동네)는 규칙 기반 자동 추출로 오추출이 있을 수 있고, 검색
결과의 관련성은 통계적 유사도와 AI 선별로 계산된 것으로 내용의 정확성·최신성·
적법성을 보증하지 않습니다. 정확한 내용은 각 카드의 원문 링크에서 확인하십시오.
