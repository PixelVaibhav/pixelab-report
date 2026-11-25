"""Pydantic schemas that validate everything the AI model returns."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator

KINDS = ("lab", "finding", "medication")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _txt(v):
    return None if v is None or str(v).strip().lower() in ("", "null", "none") else str(v)


class Item(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: Literal["lab", "finding", "medication"] = "lab"
    name: str
    value_text: Optional[str] = None
    value_number: Optional[float] = None
    unit: Optional[str] = None
    ref_text: Optional[str] = None
    ref_min: Optional[float] = None
    ref_max: Optional[float] = None
    flag: Literal["normal", "abnormal", "unclear"] = "unclear"
    box_2d: Optional[list[float]] = None

    @field_validator("value_number", "ref_min", "ref_max", mode="before")
    @classmethod
    def _n(cls, v):
        return _num(v)

    @field_validator("value_text", "unit", "ref_text", mode="before")
    @classmethod
    def _s(cls, v):
        return _txt(v)

    @field_validator("kind", mode="before")
    @classmethod
    def _k(cls, v):
        return v if v in KINDS else "lab"

    @field_validator("flag", mode="before")
    @classmethod
    def _f(cls, v):
        v = str(v).lower()
        return v if v in ("normal", "abnormal", "unclear") else "unclear"

    @field_validator("box_2d", mode="before")
    @classmethod
    def _b(cls, v):
        try:
            b = [float(x) for x in v]
            return b if len(b) == 4 else None
        except Exception:
            return None

    @field_validator("name", mode="before")
    @classmethod
    def _name(cls, v):
        s = str(v or "").strip()
        if not s:
            raise ValueError("empty name")
        return s


class PageResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    page_type: str = "other"
    title: Optional[str] = None
    modality: Optional[str] = None
    body_part: Optional[str] = None
    scan_description: Optional[str] = None
    items: list[Item] = []

    @field_validator("title", "modality", "body_part", "scan_description", mode="before")
    @classmethod
    def _t(cls, v):
        return _txt(v)


def parse_page(data):
    """Validate one page of model output. Bad items are dropped and counted, not fatal."""
    if isinstance(data, list):
        data = {"items": data}
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    items, bad = [], 0
    for raw in data.get("items") or []:
        try:
            items.append(Item.model_validate(raw))
        except Exception:
            bad += 1
    head = {k: data.get(k) for k in ("page_type", "title", "modality", "body_part", "scan_description")}
    head["page_type"] = str(head["page_type"] or "other")
    return PageResult(**head, items=items), bad
