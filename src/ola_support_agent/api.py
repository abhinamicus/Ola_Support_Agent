from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .crew import BudgetExceededError, OlaSupportCrew
from .schemas import AddDocumentRequest, AskRequest, AskResponse

app = FastAPI(title="Ola Domain Support Agent")
crew = OlaSupportCrew()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return AskResponse(response=crew.kickoff(request.query, session_id=request.session_id))


@app.post("/add-document")
def add_document(request: AddDocumentRequest) -> dict[str, str]:
    target = (
        Path(__file__).resolve().parents[2]
        / "knowledge_base"
        / f"{request.document_id}.md"
    )

    target.write_text(
        request.content.strip() + "\n",
        encoding="utf-8",
    )

    refreshed_crew = OlaSupportCrew()
    crew.rag = refreshed_crew.rag
    crew.rag_tool._rag = crew.rag

    return {
        "status": "accepted",
        "document_id": request.document_id,
        "note": "document added and index reloaded",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        while True:
            payload = await websocket.receive_json()

            try:
                request = AskRequest.model_validate(payload)
            except ValueError as exc:
                await websocket.send_json(
                    {
                        "error": "Invalid request payload",
                        "details": str(exc),
                    }
                )
                continue

            try:
                response = crew.kickoff(
                    request.query,
                    session_id=request.session_id,
                )
                await websocket.send_json(response.model_dump())

            except BudgetExceededError as exc:
                await websocket.send_json({"error": str(exc)})

    except WebSocketDisconnect:
        return
