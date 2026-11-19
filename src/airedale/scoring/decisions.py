# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""OpenAI Decisions API evaluator for opt-in binary rubric grading."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import openai
from ddtrace.llmobs import BaseAsyncEvaluator, EvaluatorContext, EvaluatorResult

from airedale.gateway import resolve_provider_config
from airedale.scoring.rubric import _extract_answer, _extract_criteria, _extract_question

if TYPE_CHECKING:
    from airedale.config.gateway import GatewayConfig

DECISIONS_MODEL = "gpt-6-luna"
MAX_QUESTIONS = 400


class DecisionsRubricEvaluator(BaseAsyncEvaluator):
    """Score criteria with predicate decisions; invalid responses fail safely."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        gateway: GatewayConfig | None = None,
        openai_client: openai.AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(name="decisions_rubric_score")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("Decisions threshold must be a number between 0 and 1")
        self.threshold = threshold
        self.gateway = gateway
        self._client = openai_client

    @property
    def client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            # Do not send prompts or gateway headers to a redirected origin.
            kwargs: dict[str, Any] = {"http_client": openai.DefaultAsyncHttpxClient(follow_redirects=False)}
            if self.gateway is not None:
                resolved = resolve_provider_config("openai", self.gateway)
                kwargs["base_url"] = resolved.base_url
                kwargs["api_key"] = "sk-not-used" if resolved.bearer_token else resolved.api_key
                headers = dict(resolved.headers)
                if resolved.bearer_token:
                    headers.setdefault("Authorization", f"Bearer {resolved.bearer_token}")
                if headers:
                    kwargs["default_headers"] = headers
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def evaluate(self, context: EvaluatorContext) -> EvaluatorResult:
        criteria = _extract_criteria(context.expected_output)
        if not criteria:
            return EvaluatorResult(value=0.0, assessment="fail", reasoning="No rubric criteria were provided.")
        if len(criteria) > MAX_QUESTIONS:
            return self._failed(criteria, "question_limit")
        response = await self._request(context, criteria)
        try:
            payload = response.http_response.json()
        except (ValueError, TypeError):
            return self._failed(criteria, "invalid_response")
        answers = payload.get("answers") if isinstance(payload, dict) else None
        if not isinstance(answers, list) or len(answers) != len(criteria):
            return self._failed(criteria, "invalid_response")
        results: list[dict[str, Any]] = []
        for index, (criterion, answer) in enumerate(zip(criteria, answers, strict=True)):
            if not isinstance(answer, dict) or answer.get("name") != str(index):
                results.append({"criterion": criterion, "score": 0, "probability": None, "status": "invalid"})
                continue
            probability = answer.get("probability")
            answer_type = answer.get("type")
            if answer_type == "refusal":
                results.append({"criterion": criterion, "score": 0, "probability": None, "status": "refusal"})
            elif (
                answer_type != "predicate"
                or isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not 0 <= probability <= 1
            ):
                results.append({"criterion": criterion, "score": 0, "probability": None, "status": "invalid"})
            else:
                results.append(
                    {
                        "criterion": criterion,
                        "score": int(probability >= self.threshold),
                        "probability": probability,
                        "status": "ok",
                    }
                )
        return self._result(results)

    async def _request(self, context: EvaluatorContext, criteria: list[str]):
        try:
            return await self.client.decisions.with_raw_response.create(
                model=DECISIONS_MODEL,
                input=f"Question:\n{_extract_question(context.input_data)}\n\nAnswer:\n{_extract_answer(context.output_data)}",
                questions=[
                    {"name": str(index), "type": "predicate", "instructions": criterion}
                    for index, criterion in enumerate(criteria)
                ],
            )
        except Exception:
            pass
        # Raise outside the handler so the provider exception is not chained.
        raise RuntimeError("Decisions provider request failed")

    def _failed(self, criteria: list[str], status: str) -> EvaluatorResult:
        return self._result(
            [{"criterion": criterion, "score": 0, "probability": None, "status": status} for criterion in criteria]
        )

    def _result(self, results: list[dict[str, Any]]) -> EvaluatorResult:
        score = sum(result["score"] for result in results) / len(results)
        invalid_count = sum(result["status"] not in {"ok", "refusal"} for result in results)
        refusal_count = sum(result["status"] == "refusal" for result in results)
        return EvaluatorResult(
            value=score,
            assessment="pass" if score >= 0.8 and not invalid_count and not refusal_count else "fail",
            reasoning=json.dumps(results),
            metadata={
                "grader": "decisions",
                "model": DECISIONS_MODEL,
                "threshold": self.threshold,
                "criteria_results": results,
                "refusal_count": refusal_count,
                "invalid_answer_count": invalid_count,
            },
            tags={"grader": "decisions", "judge_model": DECISIONS_MODEL},
        )
