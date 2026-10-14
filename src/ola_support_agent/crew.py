from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

from .guards import apply_input_guardrails
from .logging_utils import JsonlLogger
from .memory import SessionMemory
from .mock_llm import MockLLM
from .rag import RAGEngine
from .schemas import CrewResponse
from .tickets import check_support_ticket_status

from pydantic import BaseModel, PrivateAttr
from crewai.utilities.converter import Converter


class BudgetExceededError(ValueError):
    pass



class SchemaAwareConverter(Converter):
    """Validate structured JSON already produced by the composer."""

    def to_pydantic(self, current_attempt: int = 1) -> BaseModel:
        decoder = json.JSONDecoder()

        for match in re.finditer(r"\{", self.text):
            try:
                obj, _ = decoder.raw_decode(self.text[match.start():])
            except json.JSONDecodeError:
                continue

            if not isinstance(obj, dict):
                continue

            required_fields = {
                "trace_id",
                "user_query",
                "sanitized_query",
                "answer",
                "sources",
                "lookup_result",
                "grounded",
            }

            if required_fields.issubset(obj):
                return self.model.model_validate(obj)

        raise ValueError(
            "Composer output did not contain a valid structured response."
        )



class RAGTool(BaseTool):
    name: str = "retrieve_ola_support_knowledge"
    description: str = (
        "Retrieve grounded information from the approved Ola support"
        "knowledge base. Input is a support question."
    )
    _rag: Any = PrivateAttr()

    def __init__(self, rag: RAGEngine, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._rag = rag

    def _run(self, query: str) -> str:
        result = self._rag.grounded_answer(
            query,
            strategy="sentence",
        )

        output = result.model_dump_json()

        print(">>> ACTUAL RAG TOOL RESULT:")
        print(output)

        return output


class TicketLookupTool(BaseTool):
    name: str = "check_support_ticket_status"
    description: str = (
        "Look up a support ticket by record ID such as OLA-0001. "
        "Returns status, resolution time and escalation information."
    )

    def _run(self, record_id: str) -> str:
        result = check_support_ticket_status(record_id)

        output = result.model_dump_json()

        print(">>> ACTUAL LOOKUP TOOL RESULT:")
        print(output)

        return output
    


class OlaSupportCrew:
    """
    Public application wrapper around a real CrewAI Crew.

    The wrapper keeps the existing application-facing kickoff() contract
    while delegating the agent workflow to CrewAI.
    """

    def __init__(self, rag_engine: RAGEngine | None = None) -> None:
        # Must happen before CrewAI objects are initialized.
        os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

        self.rag = rag_engine or RAGEngine()
        self.memory = SessionMemory()
        self.logger = JsonlLogger()
    

        self.request_budget_chars = 650

        # Governance invariant:
        # only lookup_agent may receive the lookup tool.
        self.lookup_tool_enabled_agents = {"lookup_agent"}

        self.llm = MockLLM()

        self.rag_tool = RAGTool(rag=self.rag)
        self.lookup_tool = TicketLookupTool()

        self.retrieval_agent = Agent(
            role="Retrieval Agent",
            goal=(
                "Retrieve the most relevant approved Ola support knowledge "
                "for the user's question."
            ),
            backstory=(
                "You are the knowledge retrieval specialist. You only use "
                "the approved support knowledge base."
            ),
            tools=[self.rag_tool],
            llm=self.llm,
            allow_delegation=False,
            verbose=False,
            max_iter=2,
        )

        self.lookup_agent = Agent(
            role="Lookup Agent",
            goal=(
                "Retrieve factual support-ticket status information when "
                "a ticket identifier is supplied."
            ),
            backstory=(
                "You are the ticket lookup specialist. You use only the "
                "support-ticket lookup tool."
            ),
            tools=[self.lookup_tool],
            llm=self.llm,
            allow_delegation=False,
            verbose=False,
            max_iter=2,
        )

        self.composer_agent = Agent(
            role="Response Composer",
            goal=(
                "Compose a concise, grounded support response from the "
                "retrieval and ticket lookup results."
            ),
            backstory=(
                "You are the final response specialist. You must not "
                "invent information or call external tools."
            ),
            tools=[],
            llm=self.llm,
            allow_delegation=False,
            verbose=False,
            max_iter=1,
        )

        self._assert_tool_governance()

    def _assert_tool_governance(self) -> None:
        """
        Fail closed if the lookup tool is accidentally wired to another
        agent.
        """

        lookup_name = self.lookup_tool.name

        for agent in (
            self.retrieval_agent,
            self.composer_agent,
        ):
            for tool in agent.tools:
                if getattr(tool, "name", "") == lookup_name:
                    raise RuntimeError(
                        f"Governance violation: {lookup_name} is assigned "
                        f"to {agent.role}."
                    )

        lookup_names = {
            getattr(tool, "name", "")
            for tool in self.lookup_agent.tools
        }

        if lookup_name not in lookup_names:
            raise RuntimeError(
                "Governance violation: lookup_agent does not have "
                "the ticket lookup tool."
            )

    def _enforce_budget(self, text: str) -> None:
        if len(text) > self.request_budget_chars:
            raise BudgetExceededError(
                f"Rejected oversized request with {len(text)} chars; "
                f"budget cap is {self.request_budget_chars}."
            )

    @staticmethod
    def _contains_ticket_id(query: str) -> bool:
        return re.search(
            r"\bOLA-\d{4}\b",
            query,
            flags=re.IGNORECASE,
        ) is not None

    @staticmethod
    def _needs_support_knowledge(query: str) -> bool:
        """Identify requests that need support-policy retrieval."""
        lowered = query.lower()

        knowledge_indicators = (
            "policy",
            "procedure",
            "protocol",
            "sla",
            "first response",
            "escalat",
            "refund",
            "compensation",
            "public holiday",
            "business hours",
            "retention",
            "retain",
            "vip",
            "repeat complaint",
            "service credit",
            "communication channel",
            "feedback",
            "knowledge base",
        )

        return any(term in lowered for term in knowledge_indicators)

    def _memory_answer(
        self,
        query: str,
        session_id: str,
    ) -> str | None:
        lowered = query.lower()

        if (
            "what did i ask" not in lowered
            and "same customer" not in lowered
        ):
            return None

        history = self.memory.history(session_id)

        if not history:
            return "There is no prior conversation state for this session yet."

        previous_user_turns = [
            item["content"]
            for item in history
            if item["role"] == "user"
        ]

        if "same customer" in lowered and previous_user_turns:
            return (
                "Using the same session context, your earlier request was: "
                f"{previous_user_turns[-1]}"
            )

        if previous_user_turns:
            return (
                "Earlier in this session you asked: "
                f"{previous_user_turns[-1]}"
            )

        return "There is no prior user message in this session yet."

    def _build_crew(
        self,
        query: str,
        session_id: str,
    ) -> Crew:
        has_ticket_id = self._contains_ticket_id(query)
        needs_knowledge = (
            not has_ticket_id or self._needs_support_knowledge(query)
        )

        tasks = []

        if needs_knowledge:
            retrieval_task = Task(
                name="retrieve_support_context",
                description=(
                    "Retrieve approved support knowledge for this user query:\n"
                    f"{query}"
                ),
                expected_output=(
                    "The JSON result returned by the retrieval tool, including "
                    "groundedness, retrieved chunks and matched documents."
                ),
                agent=self.retrieval_agent,
            )
            tasks.append(retrieval_task)

        if has_ticket_id:
            lookup_task = Task(
                name="lookup_support_ticket",
                description=(
                    "Use the ticket lookup tool to retrieve factual ticket "
                    "information for the supplied ticket ID.\n\n"
                    f"User query: {query}"
                ),
                expected_output=(
                    "The JSON result returned by the support-ticket lookup "
                    "tool."
                ),
                agent=self.lookup_agent,
            )

            if needs_knowledge:
                lookup_task.context = [retrieval_task]

            tasks.append(lookup_task)

        composer_task = Task(
            name="compose_support_response",
            description=(
                "Compose the final answer using only the preceding agent "
                "results. Address the user's actual request.\n\n"
                f"User query: {query}\n"
                f"Session: {session_id}\n\n"
                "For ticket-only requests, use the ticket lookup result and "
                "do not require policy retrieval. For requests needing support "
                "knowledge, use the retrieval result. For mixed requests, use "
                "both results. Do not invent facts. If required knowledge is "
                "not grounded, clearly say that the approved knowledge base "
                "does not contain the answer."
            ),
            expected_output=(
                "A concise final customer-support answer grounded in the "
                "provided agent results."
            ),
            agent=self.composer_agent,
            context=tasks.copy(),
            output_pydantic=CrewResponse,
            converter_cls=SchemaAwareConverter,
        )

        tasks.append(composer_task)

        return Crew(
            agents=[
                self.retrieval_agent,
                self.lookup_agent,
                self.composer_agent,
            ],
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )

    def kickoff(
        self,
        query: str,
        session_id: str = "default",
    ) -> CrewResponse:
        started = time.perf_counter()
        trace_id = self.logger.new_trace_id()

        try:
            self._enforce_budget(query)

            guard = apply_input_guardrails(query)

            if guard.blocked:
                response = CrewResponse(
                    trace_id=trace_id,
                    user_query=query,
                    sanitized_query=guard.sanitized_text,
                    answer=guard.reason or "Request blocked.",
                    sources=[],
                    grounded=False,
                    guardrail_events=guard.events,
                )

                self.logger.log(
                    trace_id=trace_id,
                    duration_ms=round(
                        (time.perf_counter() - started) * 1000,
                        2,
                    ),
                    query=guard.sanitized_text,
                    guardrail_events=guard.events,
                    grounded=False,
                )

                return response

            memory_answer = self._memory_answer(
                guard.sanitized_text,
                session_id,
            )

            # Memory turns don't need to invoke the knowledge agents.
            if memory_answer is not None:
                response = CrewResponse(
                    trace_id=trace_id,
                    user_query=query,
                    sanitized_query=guard.sanitized_text,
                    answer=memory_answer,
                    sources=[],
                    grounded=True,
                    guardrail_events=guard.events,
                    review_applied=False,
                )

                self.memory.append(
                    session_id,
                    "user",
                    guard.sanitized_text,
                )
                self.memory.append(
                    session_id,
                    "assistant",
                    response.answer,
                )

                self.logger.log(
                    trace_id=trace_id,
                    duration_ms=round(
                        (time.perf_counter() - started) * 1000,
                        2,
                    ),
                    query=guard.sanitized_text,
                    guardrail_events=guard.events,
                    grounded=True,
                    cache_hit=False,
                    sources=[],
                )

                return response

            crew = self._build_crew(
                guard.sanitized_text,
                session_id,
            )

            print("QUERY SENT TO CREW:", guard.sanitized_text)
            print("CREW TASKS:", [task.name for task in crew.tasks])

            crew_result = crew.kickoff()

            # Prefer CrewAI's structured Pydantic output.
            candidate = getattr(crew_result, "pydantic", None)

            # Fall back to the raw result if structured output is unavailable.
            if candidate is None:
                candidate = getattr(crew_result, "raw", str(crew_result))

            # Validate at the application boundary, regardless of the output path.
            try:
                if isinstance(candidate, str):
                    response = CrewResponse.model_validate_json(candidate)
                else:
                    response = CrewResponse.model_validate(candidate)
            except Exception:
                response = CrewResponse(
                    trace_id=trace_id,
                    user_query=query,
                    sanitized_query=guard.sanitized_text,
                    answer=(
                        "I could not verify a grounded answer from the approved "
                        "Ola support knowledge base."
                    ),
                    sources=[],
                    grounded=False,
                    guardrail_events=guard.events,
                    review_applied=False,
                )
            # Enforce application-owned identity fields after CrewAI output.
            response = response.model_copy(
                update={
                    "trace_id": trace_id,
                    "user_query": query,
                    "sanitized_query": guard.sanitized_text,
                    "guardrail_events": guard.events,
                    "review_applied": False,
                }
            )

            self.memory.append(
                session_id,
                "user",
                guard.sanitized_text,
            )
            self.memory.append(
                session_id,
                "assistant",
                response.answer,
            )

            self.logger.log(
                trace_id=trace_id,
                duration_ms=round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                query=guard.sanitized_text,
                guardrail_events=guard.events,
                grounded=response.grounded,
                cache_hit=response.cache_hit,
                sources=response.sources,
            )

            return response

        except BudgetExceededError:
            # Still create a request trace before re-raising.
            self.logger.log(
                trace_id=trace_id,
                duration_ms=round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                query=query,
                guardrail_events=["budget_exceeded"],
                grounded=False,
            )
            raise
