# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Per-criterion LLM-as-judge rubric evaluator for LLMObs experiments."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anthropic
import openai
from ddtrace.llmobs import BaseAsyncEvaluator, EvaluatorContext, EvaluatorResult

from airedale.gateway import resolve_provider_config
from airedale.types import ModelSpec

if TYPE_CHECKING:
    from airedale.config.gateway import GatewayConfig

DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4-6"

_JUDGE_SYSTEM_PROMPT = "You are a strict technical-answer evaluator. Return only JSON."


@dataclass(frozen=True)
class CriterionJudgement:
    """Result for one rubric criterion."""

    criterion: str
    score: float
    reasoning: str


class RubricEvaluator(BaseAsyncEvaluator):
    """Evaluate each rubric criterion with an independent judge call.

    A task's list of criteria is decomposed into one criterion per judge call.
    The final row score is the arithmetic mean normalized to ``0..1`` and the
    assessment passes when the mean meets ``pass_threshold``. The judge model is
    independent of the scenario model under evaluation.

    When a :class:`GatewayConfig` is supplied, the judge clients honor it: the
    async client for the judge's provider is built lazily with the resolved
    ``base_url``/``api_key``/``default_headers`` so judge traffic flows through
    the same gateway as the scenario runs. Without a gateway the standard
    provider SDK defaults (env-var API keys) are used.
    """

    def __init__(
        self,
        *,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        gateway: GatewayConfig | None = None,
        pass_threshold: float = 0.8,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        openai_client: openai.AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(name="rubric_score")
        self.model = ModelSpec.parse(judge_model)
        self.gateway = gateway
        self.pass_threshold = pass_threshold
        self._anthropic_client = anthropic_client
        self._openai_client = openai_client

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """Return the Anthropic client (backwards-compatible alias)."""
        return self.anthropic_client

    @property
    def anthropic_client(self) -> anthropic.AsyncAnthropic:
        """Create the Anthropic judge client lazily, honoring the gateway."""
        if self._anthropic_client is None:
            base_url, api_key, headers = self._resolve_client_kwargs("anthropic")
            kwargs: dict[str, Any] = {}
            if base_url:
                kwargs["base_url"] = base_url
            if api_key:
                kwargs["api_key"] = api_key
            if headers:
                kwargs["default_headers"] = headers
            self._anthropic_client = anthropic.AsyncAnthropic(**kwargs)
        return self._anthropic_client

    @property
    def openai_client(self) -> openai.AsyncOpenAI:
        """Create the OpenAI judge client lazily, honoring the gateway."""
        if self._openai_client is None:
            base_url, api_key, headers = self._resolve_client_kwargs("openai")
            kwargs: dict[str, Any] = {}
            if base_url:
                kwargs["base_url"] = base_url
            if api_key:
                kwargs["api_key"] = api_key
            if headers:
                kwargs["default_headers"] = headers
            self._openai_client = openai.AsyncOpenAI(**kwargs)
        return self._openai_client

    def _resolve_client_kwargs(self, provider: str) -> tuple[str | None, str | None, dict[str, str]]:
        """Resolve (base_url, api_key, default_headers) for the judge provider.

        When the gateway injects a bearer token we add an ``Authorization`` header
        and pass a placeholder ``api_key`` (the SDKs require a non-empty key even
        though the gateway authenticates via the header). When no gateway is
        configured this returns ``(None, None, {})`` so the SDK falls back to its
        standard environment-variable credentials.
        """
        if self.gateway is None:
            return None, None, {}
        resolved = resolve_provider_config(provider, self.gateway)
        headers = dict(resolved.headers)
        if resolved.bearer_token:
            headers.setdefault("Authorization", f"Bearer {resolved.bearer_token}")
            api_key = "sk-not-used"
        else:
            api_key = resolved.api_key
        return resolved.base_url, api_key, headers

    async def evaluate(self, context: EvaluatorContext) -> EvaluatorResult:
        """Score the output against all criteria from ``expected_output``."""
        criteria = _extract_criteria(context.expected_output)
        answer = _extract_answer(context.output_data)
        question = _extract_question(context.input_data)
        if not criteria:
            return EvaluatorResult(value=0.0, assessment="fail", reasoning="No rubric criteria were provided.")

        judgements = await asyncio.gather(
            *(self._judge_criterion(question=question, answer=answer, criterion=criterion) for criterion in criteria)
        )
        score = sum(judgement.score for judgement in judgements) / len(judgements)
        reasoning = json.dumps(
            [
                {"criterion": judgement.criterion, "score": judgement.score, "reasoning": judgement.reasoning}
                for judgement in judgements
            ],
            ensure_ascii=False,
        )
        return EvaluatorResult(
            value=score,
            assessment="pass" if score >= self.pass_threshold else "fail",
            reasoning=reasoning,
            metadata={
                "judge_model": self.model.label,
                "criteria_count": len(criteria),
                "criteria_results": [judgement.__dict__ for judgement in judgements],
            },
            tags={"judge_model": self.model.label},
        )

    async def _judge_criterion(self, *, question: str, answer: str, criterion: str) -> CriterionJudgement:
        prompt = _judge_prompt(question=question, answer=answer, criterion=criterion)
        if self.model.provider == "openai":
            text = await self._judge_with_openai(prompt)
        else:
            text = await self._judge_with_anthropic(prompt)
        parsed = _parse_judge_json(text)
        return CriterionJudgement(
            criterion=criterion,
            score=_normalize_score(parsed.get("score")),
            reasoning=str(parsed.get("reasoning") or text),
        )

    async def _judge_with_anthropic(self, prompt: str) -> str:
        response = await self.anthropic_client.messages.create(
            model=self.model.name,
            max_tokens=1024,
            temperature=0,
            system=_JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "\n".join(str(getattr(block, "text", "")) for block in getattr(response, "content", [])).strip()

    async def _judge_with_openai(self, prompt: str) -> str:
        response = await self.openai_client.chat.completions.create(
            model=self.model.name,
            max_tokens=1024,
            temperature=0,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        choices = getattr(response, "choices", []) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "")
        if isinstance(content, list):
            return "\n".join(str(getattr(block, "text", block)) for block in content).strip()
        return str(content or "").strip()


def _judge_prompt(*, question: str, answer: str, criterion: str) -> str:
    return f"""
Evaluate whether the answer satisfies exactly one rubric criterion.

Question:
{question}

Answer:
{answer}

Criterion:
{criterion}

Return JSON with this exact shape:
{{"score": <number between 0 and 1>, "reasoning": "<short explanation>"}}
A score of 1 means the criterion is fully satisfied, 0.5 partially satisfied,
and 0 not satisfied. Do not evaluate any criterion except the one shown above.
""".strip()


def _extract_criteria(expected_output: Any) -> list[str]:
    criteria = expected_output.get("criteria") if isinstance(expected_output, dict) else expected_output
    if isinstance(criteria, list):
        return [str(item).strip() for item in criteria if str(item).strip()]
    if isinstance(criteria, str) and criteria.strip():
        return [criteria.strip()]
    return []


def _extract_answer(output_data: Any) -> str:
    if isinstance(output_data, dict):
        answer = output_data.get("answer")
        if answer is not None:
            return str(answer)
    return str(output_data or "")


def _extract_question(input_data: Any) -> str:
    if isinstance(input_data, dict):
        prompt = input_data.get("prompt")
        if prompt is not None:
            return str(prompt)
    return str(input_data or "")


def _parse_judge_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"score": 0, "reasoning": text}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"score": 0, "reasoning": text}
    return parsed if isinstance(parsed, dict) else {"score": 0, "reasoning": text}


def _normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))
