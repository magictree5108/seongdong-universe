"""성동 유니버스 온톨로지 스키마 — 팔란티어 '의미 계층(semantic layer)'의 경량 재현.

「팔란티어 온톨로지 심층 조사 및 성동 유니버스 개발 가이드라인」의 설계 제안을
그대로 따른다: 8개 Object Type + 8개 Link Type. 팔란티어의 Action Type(쓰기/실행)
계층은 의도적으로 구현하지 않는다 — 읽기 전용 의미 계층만 재현한다(안전 원칙 4).

US 7,962,495(동적 온톨로지)의 교훈대로 스키마는 코드로 버전 관리하며,
성동구 조례·사업 구조가 바뀌면 이 파일을 편집·확장한다.

객체 필드는 영문 snake_case로 정의하고, 한국어 스펙 명칭은 각 필드의
description에 명시한다. 모든 데이터는 공개 채널(elis.go.kr, sd.go.kr,
서울재정포털 등)에서만 수집한다(안전 원칙 1).
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.2.0"


# ── Object Types (8) ─────────────────────────────────────────────


class SDObject(BaseModel):
    """모든 객체 타입의 공통 골격. 객체는 고유 식별자를 가진 1급 개체다."""

    id: str = Field(description="고유 식별자 (예: ordinance:11200101000001)")
    name: str = Field(description="대표 표시명")
    source_label: Optional[str] = Field(None, description="출처 표시명 (공개 채널)")
    source_url: Optional[str] = Field(None, description="원문 URL")


class Policy(SDObject):
    """정책/사업 — 온톨로지의 중심 객체. 예산·조례·보도자료가 여기에 연결된다."""

    type: Literal["Policy"] = "Policy"
    department: Optional[str] = Field(None, description="소관부서")
    dept_code: Optional[str] = Field(
        None, description="부서코드 (지방재정365 dept_cd — 부서명 정합은 3단계)"
    )
    field: Optional[str] = Field(None, description="분야 (예: 주택, 환경, 복지)")
    budget_current: Optional[int] = Field(None, description="예산현액 (원)")
    expenditure: Optional[int] = Field(None, description="지출액 (원)")
    basis_ordinance: Optional[str] = Field(None, description="근거조례명")
    year: Optional[int] = Field(None, description="연도")


class Department(SDObject):
    """부서 — 성동구 조직도의 과·담당관 단위. 개인 식별정보는 담지 않는다(안전 원칙 2)."""

    type: Literal["Department"] = "Department"
    parent_org: Optional[str] = Field(None, description="상위조직 (국·단)")
    duties: Optional[str] = Field(None, description="소관업무 (업무분장, 조직 단위)")


class Ordinance(SDObject):
    """조례/자치법규 — name이 법규명."""

    type: Literal["Ordinance"] = "Ordinance"
    kind: Optional[str] = Field(None, description="종류 (조례/규칙/훈령규정 등)")
    department: Optional[str] = Field(None, description="소관부서")
    full_text: Optional[str] = Field(None, description="조문 전문")
    revision_history: Optional[str] = Field(None, description="개정이력")
    year: Optional[int] = Field(None, description="최종 개정 연도")


class BudgetItem(SDObject):
    """예산항목 — 세부사업 단위의 예산. name이 세부사업명."""

    type: Literal["BudgetItem"] = "BudgetItem"
    account: Optional[str] = Field(None, description="회계구분 (일반/특별)")
    field: Optional[str] = Field(None, description="분야")
    budget_current: Optional[int] = Field(None, description="예산현액 (원)")
    expenditure: Optional[int] = Field(None, description="지출액 (원)")
    balance: Optional[int] = Field(None, description="집행잔액 (원)")
    year: Optional[int] = Field(None, description="회계연도")


class PressRelease(SDObject):
    """보도자료 — name이 제목. 새소식(news)도 같은 타입으로 승격하고 subtype으로 구분."""

    type: Literal["PressRelease"] = "PressRelease"
    subtype: Optional[str] = Field(None, description="보도자료/새소식")
    published_at: Optional[str] = Field(None, description="등록일 (YYYY-MM-DD)")
    body: Optional[str] = Field(None, description="본문")
    mentioned_policies: list[str] = Field(
        default_factory=list, description="언급사업 (3단계에서 Claude로 추출)"
    )


class Facility(SDObject):
    """시설 — 공공 시설물. name이 시설명."""

    type: Literal["Facility"] = "Facility"
    kind: Optional[str] = Field(None, description="유형 (도서관/체육시설 등)")
    address: Optional[str] = Field(None, description="주소")
    lat: Optional[float] = Field(None, description="위도")
    lng: Optional[float] = Field(None, description="경도")
    department: Optional[str] = Field(None, description="소관부서")


class District(SDObject):
    """행정동 — name이 동명."""

    type: Literal["District"] = "District"
    population: Optional[int] = Field(None, description="인구")
    jurisdiction: Optional[str] = Field(None, description="관할 (법정동 등)")


class ComplaintType(SDObject):
    """민원유형 — 공개 통계 범위 내 유형만. 개인 민원 원문은 배제한다(안전 원칙 2)."""

    type: Literal["ComplaintType"] = "ComplaintType"
    department: Optional[str] = Field(None, description="소관부서")


class NationalLaw(SDObject):
    """국가법령 — 조례가 인용하는 상위법 (법제처 국가법령정보 API로 실체화).

    조례 원문의 「」 인용에서 발견된 법령을 law.go.kr에서 조회해 만든다.
    부서→사업→조례→국가법령으로 이어지는 법적 근거 사슬의 마지막 칸.
    (스키마 1.2.0에서 추가 — 원 설계서 8객체의 확장)"""

    type: Literal["NationalLaw"] = "NationalLaw"
    kind: Optional[str] = Field(None, description="법령구분 (법률/대통령령/부령 등)")
    ministry: Optional[str] = Field(None, description="소관부처")
    promulgated_at: Optional[str] = Field(None, description="공포일자 (YYYY-MM-DD)")
    effective_at: Optional[str] = Field(None, description="시행일자 (YYYY-MM-DD)")
    law_id: Optional[str] = Field(None, description="법제처 법령ID")
    cited_count: Optional[int] = Field(None, description="성동구 조례에서 인용된 횟수")


AnyObject = Union[
    Policy, Department, Ordinance, BudgetItem,
    PressRelease, Facility, District, ComplaintType, NationalLaw,
]

TYPE_REGISTRY: dict[str, type[SDObject]] = {
    "Policy": Policy,
    "Department": Department,
    "Ordinance": Ordinance,
    "BudgetItem": BudgetItem,
    "PressRelease": PressRelease,
    "Facility": Facility,
    "District": District,
    "ComplaintType": ComplaintType,
    "NationalLaw": NationalLaw,
}

TYPE_LABELS: dict[str, str] = {
    "Policy": "정책/사업",
    "Department": "부서",
    "Ordinance": "조례/자치법규",
    "BudgetItem": "예산항목",
    "PressRelease": "보도자료",
    "Facility": "시설",
    "District": "행정동",
    "ComplaintType": "민원유형",
    "NationalLaw": "국가법령",
}


# ── Link Types (8) ───────────────────────────────────────────────


class LinkType(str, Enum):
    """두 객체 타입 간 관계 — 팔란티어에서 외래키 조인에 대응하는 동사."""

    MANAGES = "담당"            # Department -담당→ Policy
    EXECUTES = "집행"           # Policy -집행→ BudgetItem
    BASIS_OF = "근거"           # Ordinance -근거→ Policy
    MENTIONS = "언급"           # PressRelease -언급→ Policy
    GOVERNS = "관할"            # Policy -관할→ District
    LOCATED_IN = "위치"         # Facility -위치→ District
    FACILITY_OWNED_BY = "시설소관"   # Facility -소관→ Department
    COMPLAINT_OWNED_BY = "민원소관"  # ComplaintType -소관→ Department
    DELEGATES = "위임"          # NationalLaw -위임→ Ordinance (조례의 상위법 근거)


# 링크 타입별 허용 (src_type, dst_type) — 삽입 시 검증에 사용
LINK_ENDPOINTS: dict[LinkType, tuple[str, str]] = {
    LinkType.MANAGES: ("Department", "Policy"),
    LinkType.EXECUTES: ("Policy", "BudgetItem"),
    LinkType.BASIS_OF: ("Ordinance", "Policy"),
    LinkType.MENTIONS: ("PressRelease", "Policy"),
    LinkType.GOVERNS: ("Policy", "District"),
    LinkType.LOCATED_IN: ("Facility", "District"),
    LinkType.FACILITY_OWNED_BY: ("Facility", "Department"),
    LinkType.COMPLAINT_OWNED_BY: ("ComplaintType", "Department"),
    LinkType.DELEGATES: ("NationalLaw", "Ordinance"),
}


class Link(BaseModel):
    """객체 간 방향 링크. 생성 근거(evidence)와 방법을 함께 저장해 검증 가능하게 한다."""

    type: LinkType
    src: str = Field(description="출발 객체 id")
    dst: str = Field(description="도착 객체 id")
    method: str = Field("rule", description="생성 방법: rule/claude/manual")
    evidence: Optional[str] = Field(None, description="링크 근거 (원문 발췌 등)")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="확신도")

    @property
    def id(self) -> str:
        return f"{self.src}|{self.type.value}|{self.dst}"
