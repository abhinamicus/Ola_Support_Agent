from __future__ import annotations
import asyncio


from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, CreateResult, RequestUsage

from .mock_llm import MockLLM
from .schemas import ReviewVerdict



class ReviewStage:
    def __init__(self) -> None:
        self.llm = MockLLM()

    def _build_model_client(self) -> ChatCompletionClient:
        """
        Adapt the existing deterministic MockLLM to AutoGen's
        ChatCompletionClient interface without introducing API keys
        or network calls.
        """
        return _MockChatCompletionClient(self.llm)

    def review(
        self,
        draft_answer: str,
        retrieved_context: str,
        *,
        allow_non_rag_answer: bool = False,
    ) -> ReviewVerdict:
        if allow_non_rag_answer:
            return ReviewVerdict(
                approved=True,
                final_answer=draft_answer,
                reason="Draft answer was produced by an approved non-RAG control path.",
            )

        model_client = self._build_model_client()

        reviewer = AssistantAgent(
            name="grounding_reviewer",
            model_client=model_client,
            system_message=(
                "You are the grounding reviewer. "
                "Inspect the draft answer against the retrieved context. "
                "Approve only if the answer is supported by the context. "
                "If unsupported, explain what must be revised."
            ),
        )

        reviser = AssistantAgent(
            name="grounding_reviser",
            model_client=model_client,
            system_message=(
                "You are the final grounding reviser. "
                "Use the reviewer's feedback and the retrieved context. "
                "Return a final grounded answer. "
                "Do not invent information."
            ),
        )

        task = (
            "Review this support response.\n\n"
            f"DRAFT ANSWER:\n{draft_answer}\n\n"
            f"RETRIEVED CONTEXT:\n{retrieved_context}\n\n"
            "Determine whether the draft is fully supported by the retrieved "
            "context. If it is supported, approve it. Otherwise revise it so "
            "that it contains only information supported by the context."
        )

        termination = MaxMessageTermination(max_messages=2)

        team = RoundRobinGroupChat(
            participants=[reviewer, reviser],
            termination_condition=termination,
            max_turns=2,
        )

        result = asyncio.run(team.run(task=task))

        return self._to_verdict(
            result=result,
            draft_answer=draft_answer,
            retrieved_context=retrieved_context,
        )

    @staticmethod
    def _to_verdict(
        *,
        result,
        draft_answer: str,
        retrieved_context: str,
    ) -> ReviewVerdict:
        """
        Convert the bounded AutoGen conversation into the application's
        Pydantic verdict schema.
        """
        messages = getattr(result, "messages", [])

        text_parts: list[str] = []

        for message in messages:
            content = getattr(message, "content", None)

            if isinstance(content, str):
                text_parts.append(content)

            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif hasattr(item, "text"):
                        text_parts.append(str(item.text))

        conversation = "\n".join(text_parts)

        normalized_draft = draft_answer.strip()
        normalized_context = retrieved_context.strip()

        if (
            normalized_draft
            and normalized_context
            and normalized_draft in normalized_context
        ):
            return ReviewVerdict(
                approved=True,
                final_answer=draft_answer,
                reason="AutoGen reviewer found the draft directly supported by retrieved context.",
            )

        # If the AutoGen conversation contains an explicit final answer,
        # use it only when it is grounded in the retrieved context.
        for candidate in reversed(text_parts):
            candidate = candidate.strip()

            if (
                candidate
                and candidate != normalized_draft
                and candidate in normalized_context
            ):
                return ReviewVerdict(
                    approved=False,
                    final_answer=candidate,
                    reason="AutoGen reviewer identified unsupported draft content and produced a grounded revision.",
                )

        # Safe deterministic fallback inside the approved review boundary.
        revised = (
            normalized_context[:450]
            if normalized_context
            else "I don't know based on the approved knowledge base."
        )

        return ReviewVerdict(
            approved=False,
            final_answer=revised,
            reason=(
                "AutoGen review completed, but the draft could not be verified "
                "as grounded; the response was constrained to retrieved context."
            ),
        )


class _MockChatCompletionClient(ChatCompletionClient):
    """
    Minimal AutoGen model client backed by the project's existing
    deterministic MockLLM.

    No network request or external API key is used.
    """

    def __init__(self, mock_llm: MockLLM) -> None:
        self.mock_llm = mock_llm

    @property
    def model_info(self) -> dict:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": "mock",
        }

    @property
    def capabilities(self) -> dict:
        return self.model_info

    async def create(
        self,
        messages,
        *,
        tools=None,
        tool_choice="auto",
        json_output=None,
        extra_create_args=None,
        cancellation_token: CancellationToken | None = None,
    ):
        prompt_parts: list[str] = []

        for message in messages:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                prompt_parts.append(content)
            else:
                prompt_parts.append(str(content))

        prompt = "\n".join(prompt_parts)

        response = self.mock_llm.call(prompt)

        from autogen_core.models import CreateResult

        return CreateResult(
            finish_reason="stop",
            content=response,
            usage=RequestUsage(
                prompt_tokens=max(1, len(prompt) // 4),
                completion_tokens=max(1, len(response) // 4),
            ),
            cached=False,
        )

    async def create_stream(
        self,
        messages,
        *,
        tools=None,
        tool_choice="auto",
        json_output=None,
        extra_create_args=None,
        cancellation_token: CancellationToken | None = None,
    ):
        result = await self.create(
            messages,
            tools=tools,
            json_output=json_output,
            extra_create_args=extra_create_args,
        )
        yield result

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def count_tokens(self, messages, *, tools=None) -> int:
        return sum(
            len(str(getattr(message, "content", "")))
            for message in messages
        )

    def remaining_tokens(self, messages, *, tools=None) -> int:
        return max(0, 8192 - self.count_tokens(messages, tools=tools))