"""NEO 성동 검색 품질 평가 하니스.

1) 시스템 내부를 모르는 '성동구 담당 공무원' 페르소나로 무작위 질의 100개 생성
   (생성 모델에게 데이터 보유 범위를 알려주지 않는다 — 선별 편향 방지)
2) 전 질의를 실제 검색 파이프라인(run_search)으로 실행
3) LLM 심사: 질의 의도 대비 결과 유용성 0~2점 + 실패 유형
   (검색실패=자료가 있을 법한데 못 찾음 / 데이터없음=보유 범위 밖 / 양호)
4) eval/report.json + 콘솔 요약

실행: ANTHROPIC_API_KEY=... .venv/bin/python eval/run_eval.py [--regen]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic
from pydantic import BaseModel

import app  # noqa: E402  (run_search·load_data 재사용)

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "questions.json"
REPORT_PATH = EVAL_DIR / "report.json"

GEN_MODEL = "claude-sonnet-5"
JUDGE_MODEL = "claude-haiku-4-5-20251001"
SEARCH_MODEL = "claude-haiku-4-5-20251001"   # 실서비스 기본값과 동일

GEN_SYSTEM = (
    "너는 서울 성동구청의 다양한 부서에서 일하는 담당 공무원 여러 명이다. "
    "구청 내부의 행정자료 검색 시스템(내용물은 모른다)에 실제로 입력할 법한 "
    "검색 질의 100개를 만들어라.\n"
    "요건:\n"
    "1. 업무 영역을 폭넓게 섞어라: 건축 인허가, 도로·주정차, 공원녹지, 환경(소음·"
    "대기·재활용), 복지(기초생활·노인·장애인·아동), 보건위생, 문화행사·축제, "
    "예산·회계·계약·입찰, 인사·조직, 세무, 민원처리, 재난·안전·민방위, 교육, "
    "청년·일자리, 주택·재개발, 교통, 반려동물, 정보공개·개인정보, 적극행정·"
    "감사·면책, 부서·담당자 찾기, 특정 동네(성수동·왕십리 등) 현안.\n"
    "2. 질의 형태도 섞어라: 짧은 키워드형(2~3단어), 자연어 질문형('~해도 되나요', "
    "'~하려면 어떤 절차가 필요한가요'), 부서명이나 동네명 단독, 법령·조례명 일부.\n"
    "3. 잘 검색될 것 같은 질의만 고르지 마라 — 실무에서 튀어나오는 구체적이고 "
    "까다로운 질의(예: 특정 상황의 허가 가능 여부, 유사 사례 찾기)를 포함하라.\n"
    "4. 100개 모두 서로 다른 주제·표현이어야 한다."
)

JUDGE_SYSTEM = (
    "너는 검색 품질 평가자다. 성동구 공무원의 질의와, 검색 시스템이 반환한 결과 "
    "목록(제목·유형)이 주어진다.\n"
    "이 시스템의 데이터 보유 범위: 성동구 고시공고(약 1,300건), 성동구 자치법규 "
    "전체(644건), 감사원·자체감사기구 사전컨설팅/적극행정면책 사례(약 800건), "
    "구 부서별 조직·업무분장(41개 부서).\n"
    "판정:\n"
    "- score 2(유용): 결과 상위가 질의 의도에 실질적으로 도움이 된다\n"
    "- score 1(부분적): 관련은 있으나 핵심을 비켜가거나 절반만 도움\n"
    "- score 0(실패): 무관한 결과이거나, 도움될 자료가 보유 범위에 있을 법한데 비어 있다\n"
    "- failure_type: '양호'(score 2) / '검색실패'(보유 범위 안일 텐데 못 찾거나 무관) "
    "/ '데이터없음'(질의 주제가 애초에 보유 범위 밖 — 이 경우 결과가 비어 있으면 "
    "정직한 동작이므로 score 1)\n"
    "reason은 한 문장."
)


class _Questions(BaseModel):
    questions: list[str]


class _Verdict(BaseModel):
    score: int
    failure_type: str
    reason: str


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(timeout=60.0, max_retries=2)


def generate_questions() -> list[str]:
    resp = _client().messages.parse(
        model=GEN_MODEL, max_tokens=8000, system=GEN_SYSTEM,
        messages=[{"role": "user", "content": "질의 100개를 생성하라."}],
        output_format=_Questions)
    qs = [q.strip() for q in resp.parsed_output.questions if q.strip()]
    if len(qs) < 90:
        raise RuntimeError(f"질의 생성 부족: {len(qs)}개")
    return qs[:100]


def run_one(q: str) -> dict:
    t0 = time.time()
    try:
        results, mode, expanded = app.run_search(q, SEARCH_MODEL)
    except Exception as exc:  # noqa: BLE001
        return {"query": q, "error": str(exc)[:200], "mode": "error",
                "results": [], "expanded": [], "elapsed": round(time.time() - t0, 1)}
    nodes, *_rest = app.load_data()
    tops = [{"title": nodes[i]["title"][:80], "category": nodes[i]["category"],
             "kind": nodes[i]["kind"]} for i, _s in results[:5]]
    return {"query": q, "mode": mode, "expanded": expanded,
            "results": tops, "n": len(results),
            "elapsed": round(time.time() - t0, 1)}


def judge_one(rec: dict) -> dict:
    listing = "\n".join(
        f"- [{r['category']}] {r['title']}" for r in rec["results"]) or "(결과 없음)"
    content = f"질의: {rec['query']}\n\n검색 결과:\n{listing}"
    try:
        resp = _client().messages.parse(
            model=JUDGE_MODEL, max_tokens=300, system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_format=_Verdict)
        v = resp.parsed_output
        return {**rec, "score": max(0, min(2, v.score)),
                "failure_type": v.failure_type, "reason": v.reason}
    except Exception as exc:  # noqa: BLE001
        return {**rec, "score": -1, "failure_type": "심사실패",
                "reason": str(exc)[:150]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="질의 100개 재생성")
    ap.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    ap.add_argument("--report", type=Path, default=REPORT_PATH)
    args = ap.parse_args()

    if args.regen or not args.questions.exists():
        print("질의 100개 생성 중 (sonnet-5)…")
        qs = generate_questions()
        args.questions.write_text(json.dumps(qs, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    qs = json.loads(args.questions.read_text(encoding="utf-8"))
    print(f"질의 {len(qs)}개 로드")

    print("검색 실행 중 (병렬 5)…")
    with ThreadPoolExecutor(max_workers=5) as ex:
        records = list(ex.map(run_one, qs))
    done = sum(1 for r in records if r["mode"] != "error")
    print(f"검색 완료: {done}/{len(records)}")

    print("심사 중 (병렬 5)…")
    with ThreadPoolExecutor(max_workers=5) as ex:
        judged = list(ex.map(judge_one, records))

    args.report.write_text(json.dumps(judged, ensure_ascii=False, indent=1),
                           encoding="utf-8")

    scores = [r["score"] for r in judged if r["score"] >= 0]
    dist = {s: scores.count(s) for s in (0, 1, 2)}
    types = {}
    for r in judged:
        types[r["failure_type"]] = types.get(r["failure_type"], 0) + 1
    print(f"\n=== 요약 ===\n점수 분포: {dist} (평균 {sum(scores)/len(scores):.2f})")
    print(f"유형: {types}")
    print(f"모드: ", {m: sum(1 for r in judged if r['mode'] == m)
                     for m in {r['mode'] for r in judged}})
    print(f"\n점수 0 (실패) {dist.get(0, 0)}건:")
    for r in judged:
        if r["score"] == 0:
            print(f"  [{r['failure_type']}] {r['query'][:40]} — {r['reason'][:60]}")
    print(f"\n리포트: {args.report}")


if __name__ == "__main__":
    main()
