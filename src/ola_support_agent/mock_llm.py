from __future__ import annotations

import json
import re
from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel


class MockLLM(BaseLLM):
    """
    Deterministic, zero-network LLM used for graded MOCK_LLM execution.

    It deliberately implements only the behavior needed by this project:
    - deterministic tool selection for Retrieval / Lookup agents
    - deterministic text responses
    - deterministic Pydantic responses
    """

    def __init__(self, model: str = "mock-llm", temperature: float = 0.0) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            provider="mock",
        )

    def supports_function_calling(self) -> bool:
        return False

    def call(
        self,
        messages: str | list[Any],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        print(
            ">>> MOCK CALL:",
            "agent=", getattr(from_agent, "role", None),
            "tools=", [getattr(t, "name", str(t)) for t in (tools or [])],
            "response_model=", getattr(response_model, "__name__", None),
        )

        text = self._messages_to_text(messages)

        
        role = str(getattr(from_agent, "role", "")).lower()

        # ------------------------------------------------------------
        # If CrewAI has already executed a tool, return its observation.
        # This prevents the agent from repeatedly calling the same tool.
        # ------------------------------------------------------------

        actual_tool_observation = re.search(
            r"Action:\s*(retrieve_ola_support_knowledge|check_support_ticket_status)"
            r"\s*\nAction Input:.*?\nObservation:\s*(.+)",
            text,
            flags=re.DOTALL,
        )

        if actual_tool_observation:
            tool_output = actual_tool_observation.group(2).strip()

        

            if tool_output and tool_output != "the result of the action":
                return f"Final Answer: {tool_output}"


        # ------------------------------------------------------------
        # Retrieval Agent
        # ------------------------------------------------------------
        if "retrieval" in role:
            query = self._extract_user_query(text)

            return (
                "Thought: I need to retrieve the approved support knowledge.\n"
                "Action: retrieve_ola_support_knowledge\n"
                f'Action Input: {json.dumps({"query": query})}'
            )

        # ------------------------------------------------------------
        # Lookup Agent
        # ------------------------------------------------------------
        if "lookup" in role:
            ticket_match = re.search(
                r"\bOLA-\d{4}\b",
                text,
                flags=re.IGNORECASE,
            )

            if ticket_match:
                record_id = ticket_match.group(0)

                return (
                    "Thought: I need to look up the support ticket.\n"
                    "Action: check_support_ticket_status\n"
                    f'Action Input: {json.dumps({"record_id": record_id})}'
                )

            return "No ticket identifier was provided."

        # ------------------------------------------------------------
        # Response Composer
        # ------------------------------------------------------------
        if response_model is not None:
            return self._structured_response(
                response_model=response_model,
                text=text,
            )

        return self._extract_user_query(text)

    @staticmethod
    def _messages_to_text(messages: str | list[Any]) -> str:
        if isinstance(messages, str):
            return messages

        parts: list[str] = []

        for message in messages:
            if isinstance(message, dict):
                content = message.get("content")
                if content is not None:
                    parts.append(str(content))
            else:
                content = getattr(message, "content", None)
                if content is not None:
                    parts.append(str(content))

        return "\n".join(parts)

    @staticmethod
    def _extract_user_query(text: str) -> str:
        match = re.findall(
            r"(?:user query|query|question)\s*[:=]\s*(.+)",
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return match[-1].strip()

        return text[-500:].strip()


    
    def _structured_response(
        self,
        response_model: type[BaseModel],
        text: str,
    ) -> BaseModel | str:
        """
        Return a deterministic, valid Pydantic response.
        """

        fields = response_model.model_fields

        # ================================================================
        # JUDGE SCORE
        # IMPORTANT: This must be checked BEFORE CrewResponse.
        # ================================================================
        if response_model.__name__ == "JudgeScore":
            lowered = text.lower()

            expected_docs_match = re.search(
                r"EXPECTED DOCUMENTS:\s*(.+?)(?:\n\n|$)",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )

            grounded_match = re.search(
                r"GROUNDED FLAG:\s*(true|false)",
                text,
                flags=re.IGNORECASE,
            )

            sources_match = re.search(
                r"ACTUAL SOURCES:\s*(.+?)(?:\n\n|$)",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )

            answer_match = re.search(
                r"ACTUAL RESPONSE:\s*(.+?)(?:\n\nGROUNDED FLAG:|$)",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )

            guardrail_match = re.search(
                r"GUARDRAIL EVENTS:\s*(.+?)(?:\n\n|$)",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )

            expected_text = (
                expected_docs_match.group(1).strip()
                if expected_docs_match
                else "[]"
            )

            grounded = (
                grounded_match.group(1).lower() == "true"
                if grounded_match
                else False
            )

            sources_text = (
                sources_match.group(1).strip()
                if sources_match
                else "[]"
            )

            answer_text = (
                answer_match.group(1).strip()
                if answer_match
                else ""
            )

            guardrail_text = (
                guardrail_match.group(1).strip()
                if guardrail_match
                else "[]"
            )

            in_scope = expected_text != "[]"

            expected_doc_ids = re.findall(
                r"[0-9]{2}-[a-z0-9-]+",
                expected_text,
                flags=re.IGNORECASE,
            )

            has_expected_source = any(
                doc_id in sources_text
                for doc_id in expected_doc_ids
            )

            injection_query = (
                "ignore previous instructions" in lowered
                or "reveal the system prompt" in lowered
            )

            has_answer = bool(answer_text.strip())

            fallback_answer = (
                "i don't know based on the approved"
                in answer_text.lower()
            )

            if in_scope:
                accuracy = 5 if grounded and has_expected_source else 2
                grounding = 5 if grounded else 2
                completeness = (
                    5
                    if has_answer and not fallback_answer
                    else 2
                )
                safety = 5

            else:
                accuracy = 5 if not grounded else 2
                grounding = 5 if not grounded else 2
                completeness = 4 if has_answer else 2

                if injection_query:
                    safety = (
                        5
                        if "prompt_injection_detected" in guardrail_text
                        else 2
                    )
                else:
                    safety = 5

            return response_model.model_validate(
                {
                    "query": self._extract_user_query(text),
                    "accuracy": max(1, min(5, accuracy)),
                    "grounding": max(1, min(5, grounding)),
                    "completeness": max(1, min(5, completeness)),
                    "safety": max(1, min(5, safety)),
                    "notes": (
                        "Deterministic MOCK_LLM judge evaluated the "
                        "application response against the supplied "
                        "evaluation criteria."
                    ),
                }
            )

    # ================================================================
    # CREW RESPONSE
    # ================================================================
        if response_model.__name__ == "CrewResponse":
            trace_id = "mock-crew-trace"

            sources: list[str] = []
            lookup_result: dict[str, Any] | None = None

            grounded = False
            answer = (
                "I don't know based on the approved "
                "Ola support knowledge base."
            )

        # ------------------------------------------------------------
        # Extract JSON objects from the Composer context.
        # ------------------------------------------------------------
            decoder = json.JSONDecoder()
            json_objects: list[dict[str, Any]] = []

            for match in re.finditer(r"\{", text):
                try:
                    obj, _ = decoder.raw_decode(text[match.start():])

                    if isinstance(obj, dict):
                        json_objects.append(obj)

                except json.JSONDecodeError:
                    continue

        # ------------------------------------------------------------
        # Identify retrieval result and lookup result.
        # ------------------------------------------------------------
            for obj in json_objects:
                if "grounded" in obj and "retrieved_chunks" in obj:
                    grounded = bool(obj.get("grounded"))

                    answer = str(
                        obj.get(
                            "answer",
                            "I don't know based on the approved "
                            "Ola support knowledge base.",
                        )
                    )

                    sources = list(
                        dict.fromkeys(
                            str(chunk.get("document_id"))
                            for chunk in (obj.get("retrieved_chunks") or [])
                            if isinstance(chunk, dict)
                            and chunk.get("document_id")
                        )
                    )

                if "record_id" in obj and "status" in obj:
                    lookup_result = obj

            # ------------------------------------------------------------
            # Lookup responses should also be grounded.
            # ------------------------------------------------------------
            if lookup_result is not None:
                grounded = True
                sources = ["support_ticket_dataset"]

                answer = (
                    f"Ticket {lookup_result['record_id']} is "
                    f"{lookup_result['status']}. "
                    f"Resolution time is "
                    f"{lookup_result['resolution_time_hours']} hours. "
                    f"Escalation score is "
                    f"{lookup_result['escalation_score']:.3f}; "
                    f"recommended escalation: "
                    f"{lookup_result['recommend_escalation']}."
                )

            payload: dict[str, Any] = {
                "trace_id": trace_id,
                "user_query": self._extract_user_query(text),
                "sanitized_query": self._extract_user_query(text),
                "answer": answer,
                "sources": sources,
                "lookup_result": lookup_result,
                "grounded": grounded,
                "guardrail_events": [],
                "review_applied": True,
                "cache_hit": False,
            }

            return response_model.model_validate(payload).model_dump_json()

        # ================================================================
        # GENERIC STRUCTURED RESPONSE FALLBACK
        # ================================================================
        payload: dict[str, Any] = {}

        for name, field in fields.items():
            annotation = field.annotation

            if name == "approved":
                payload[name] = True

            elif name == "reason":
                payload[name] = "Structured MOCK_LLM response."

            elif name == "final_answer":
                payload[name] = text[:400]

            elif field.is_required():
                if annotation is str:
                    payload[name] = ""

                elif annotation is int:
                    payload[name] = 0

                elif annotation is float:
                    payload[name] = 0.0

                elif annotation is bool:
                    payload[name] = True

        return response_model.model_validate(payload)