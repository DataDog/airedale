# Safe error text in ddtrace experiment records

## Problem and scope

`ddtrace` copies exception text from experiment tasks and evaluators into the retry summary, task span error tags, row errors, evaluator errors, failed-status text, and fail-fast output. Exception messages can contain provider responses or other private text. The retry summary is present only when retries are enabled; the other outputs can occur with the default of zero retries.

This design protects the experiment outputs. It does not change independently instrumented child spans: the agent span of `DecoratorSpanAgentRunner` and the native Claude spans can record an error before the task guard sees it. Runner errors in the process and private diagnostic reports also retain raw text.

## Guards

`experiment_task` and `RubricEvaluator.evaluate` catch `BaseException` and call `safe_error` outside the exception handler. They raise its result outside the handler as well. This prevents Python from attaching the original error as the new error's context, even if the helper itself fails.

For ordinary `Exception` instances, including `ExceptionGroup`, the helper makes an `ExperimentTaskError` or `RubricJudgeError`. Its message contains only the leaf type names and the innermost file and line, as produced by `exception_summary`. It retains the original traceback frames, not the original message or exception chain. The wrapper remains an `Exception`, so ddtrace retry and fail-fast behavior stay the same.

For a `BaseExceptionGroup` that is not an `Exception`, the helper rebuilds the group from safe children, preserving the traceback and non-`Exception` classification. A custom group with only ordinary exception leaves uses the internal `_SafeGroup` to prevent Python from converting it to an `ExceptionGroup`.

For a cancellation or another non-group `BaseException`, the helper retains the original object and message. It removes the cause and notes, and removes any context that is not a cancellation. Cancellation contexts are retained and cleaned in the same way because anyio uses their messages. The message text of these control-flow exceptions is outside the sanitization scope.

The group and cancellation walks use loops. The summary walks each group at most once, takes its first eight leaf names, skips cycles, and treats a failed custom `exceptions` property as a leaf. A parent takes only the names it needs from each child. The retained traceback contains source locations and source lines, not local values.

`DecisionsRubricEvaluator` is not wrapped: it already raises a fixed error outside its exception handler. No ddtrace configuration or default retry count changes.

## Verification

Offline tests run the real ddtrace experiment retry and span code with a fake LLM Obs service. They check that a retry summary exists before checking that it has no marker. They also check the task span, row, status text, fail-fast error, and evaluator error, along with retry counts and retained frames.

The judge tests use `httpx2.MockTransport` with an error response and set the SDK client's `max_retries=0`. They run `Experiment.run(max_retries=1)` with zero retry delay, so ddtrace controls the retry. Other tests cover cancellation chains, mixed and custom groups, the eight-leaf limit, a small cycle, and failures in the sanitization helper. The tests cover a small shared group. They do not construct deep chains or large groups.

The implementation is tested against the project's pinned ddtrace version, currently 4.17.0. Run `uv run ruff format --check .`, `uv run ruff check --no-fix .`, `uv run pyright`, and `uv run pytest` for local verification.
