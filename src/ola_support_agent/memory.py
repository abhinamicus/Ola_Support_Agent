from __future__ import annotations

from typing import Any
from collections.abc import Callable

try:
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables import RunnableLambda
    from langchain_core.runnables.history import RunnableWithMessageHistory
except Exception:  # pragma: no cover
    InMemoryChatMessageHistory = None
    AIMessage = None
    HumanMessage = None
    RunnableLambda = None
    RunnableWithMessageHistory = None


class SessionMemory:
    def __init__(self) -> None:
        self.store: dict[str, list[dict[str, str]]] = {}

    def append(self, session_id: str, role: str, content: str) -> None:
        self.store.setdefault(session_id, []).append({"role": role, "content": content})

    def history(self, session_id: str) -> list[dict[str, str]]:
        return list(self.store.get(session_id, []))


def build_langchain_history_adapter(
    invoke_fn: Callable[[dict], dict],
) -> object | None:
    if not all(
        [
            InMemoryChatMessageHistory,
            AIMessage,
            HumanMessage,
            RunnableLambda,
            RunnableWithMessageHistory,
        ]
    ):
        return None

    histories: dict[str, Any] = {}

    def get_history(session_id: str) -> Any:
        if session_id not in histories:
            histories[session_id] = InMemoryChatMessageHistory()
        return histories[session_id]

    def run_model(payload: dict) -> dict:
        result = invoke_fn(
            {
                "question": payload["question"],
                "history": payload.get("history", []),
            }
        )
        return {"answer": result["answer"]}

    runnable = RunnableLambda(run_model)

    return RunnableWithMessageHistory(
        runnable,
        get_history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="answer",
    )

