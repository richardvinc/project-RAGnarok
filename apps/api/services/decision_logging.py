from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..schemas import DecisionEvent, RunSummary


class DecisionLogger:
    def __init__(self) -> None:
        self._events: list[DecisionEvent] = []

    def log(
        self,
        *,
        step: str,
        stage: str,
        status: str,
        decision: str,
        details: dict[str, Any] | None = None,
        tool_name: str | None = None,
    ) -> None:
        self._events.append(
            DecisionEvent(
                step=step,
                stage=stage,
                status=status,  # type: ignore[arg-type]
                decision=decision,
                details=details or {},
                timestamp=datetime.now(UTC).isoformat(),
                tool_name=tool_name,
            )
        )

    @property
    def events(self) -> list[DecisionEvent]:
        return list(self._events)

    def build_summary(self) -> RunSummary:
        tool_names = [
            event.tool_name
            for event in self._events
            if event.tool_name and event.status == "completed"
        ]
        deduped = list(dict.fromkeys(tool_names))
        return RunSummary(
            used_tools=bool(deduped),
            tool_names=deduped,
            total_steps=len(self._events),
        )
