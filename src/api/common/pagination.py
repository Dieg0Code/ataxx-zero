from __future__ import annotations

from typing import TypeVar

from api.common.schemas import OffsetPage

T = TypeVar("T")


def build_page(*, items: list[T], total: int, limit: int, offset: int) -> OffsetPage[T]:
    return OffsetPage[T](
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + len(items)) < total,
    )
