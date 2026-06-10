"""otel — Hermes plugin for OpenTelemetry trace export.

Pushes session, model, tool, and cron spans to any OTLP/HTTP collector.
Fail-open: export errors are logged as warnings; no span = no crash.

Enable: HERMES_OTEL_ENABLED=1, HERMES_OTEL_ENDPOINT=http://host:port/v1/traces
Optional: HERMES_OTEL_INSECURE=1 (skip TLS), HERMES_OTEL_TIMEOUT=5
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Span kinds (OpenTelemetry semantic convention) ──────────────────────────
KIND_INTERNAL = "INTERNAL"
KIND_CLIENT = "CLIENT"
KIND_SERVER = "SERVER"
KIND_PRODUCER = "PRODUCER"
KIND_CONSUMER = "CONSUMER"

# ── Status codes ─────────────────────────────────────────────────────────────
STATUS_UNSET = 0
STATUS_OK = 1
STATUS_ERROR = 2

# ── Semantic convention attribute keys ──────────────────────────────────────
_GEN_AI_OP = "gen_ai.operation.name"
_GEN_AI_SYSTEM = "gen_ai.system"
_GEN_AI_REQ_MODEL = "gen_ai.request.model"
_GEN_AI_RESP_MODEL = "gen_ai.response.model"
_GEN_AI_USAGE_IN = "gen_ai.usage.input_tokens"
_GEN_AI_USAGE_OUT = "gen_ai.usage.output_tokens"
_GEN_AI_USAGE_TOTAL = "gen_ai.usage.total_tokens"
_GEN_AI_SESSION = "gen_ai.session.id"
_GEN_AI_EXECUTION = "gen_ai.agent.execution.id"
_GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
_GEN_AI_TOOL_NAME = "gen_ai.tool.name"

_HERMES_RESUME = "hermes.resume_from"
_HERMES_CRON_JOB = "hermes.cron.job.id"
_HERMES_PLATFORM = "hermes.platform"
_HERMES_PROVIDER = "hermes.provider"
_HERMES_MODEL = "hermes.model"
_HERMES_FINALIZE = "hermes.finalize.reason"
_HERMES_TOOL_STATUS = "hermes.tool.status"
_HERMES_TOOL_DURATION = "hermes.tool.duration_ms"
_HERMES_TOOL_ARGS_SIZE = "hermes.tool.args.size"
_HERMES_TOOL_RESULT_SIZE = "hermes.tool.result.size"
_HERMES_API_REQ_ID = "hermes.api_request_id"
_HERMES_RETRYABLE = "hermes.retryable"
_ERROR_TYPE = "error.type"


def _attr(key: str, value: Any) -> dict:
    """Build a span attribute matching OTLP JSON envelope."""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _attrs(**kwargs) -> list:
    return [_attr(k, v) for k, v in kwargs.items() if v is not None and v != ""]


def _now_ns() -> str:
    return str(int(time.time() * 1e9))


def _trace_id() -> str:
    return uuid.uuid4().hex[:32]


def _span_id() -> str:
    return uuid.uuid4().hex[:16]


# ── Persistent state (survives across hook calls within a session) ───────────
_state: dict = {}


def _reset_for_tests() -> None:
    _state.clear()


def _ensure_trace_id(session_id: str) -> str:
    key = f"trace_id:{session_id}"
    if key not in _state:
        _state[key] = _trace_id()
    return _state[key]


def _ensure_execution_id(session_id: str) -> str:
    key = f"exec_id:{session_id}"
    if key not in _state:
        _state[key] = _trace_id()
    return _state[key]


def _exporter() -> Optional[dict]:
    if not os.environ.get("HERMES_OTEL_ENABLED"):
        return None
    return {
        "endpoint": os.environ.get("HERMES_OTEL_ENDPOINT", "http://127.0.0.1:4318/v1/traces"),
        "insecure": os.environ.get("HERMES_OTEL_INSECURE", "0") == "1",
        "timeout": float(os.environ.get("HERMES_OTEL_TIMEOUT", "5")),
        "retry_interval": 60.0,
    }


# ── OTLP HTTP exporter ────────────────────────────────────────────────────────
class _Exporter:
    _last_failure: float = 0.0

    def __init__(self, config: dict) -> None:
        self.endpoint = config["endpoint"]
        self.insecure = config.get("insecure", False)
        self.timeout = config.get("timeout", 5.0)
        self.retry_interval = config.get("retry_interval", 60.0)

    def _can_send(self) -> bool:
        if self._last_failure and (time.time() - self._last_failure) < self.retry_interval:
            return False
        return True

    def export(self, spans: list) -> None:
        if not self._can_send():
            logger.warning(
                "OTEL export skipped during retry backoff; dropping %d span(s)", len(spans)
            )
            return
        payload = json.dumps({"resourceSpans": [{"schemaUrl": "", "spans": spans}]}).encode()
        try:
            import urllib.request

            req = urllib.request.Request(
                self.endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status >= 400:
                    logger.warning("OTEL collector returned HTTP %d", resp.status)
        except Exception as exc:
            self._last_failure = time.time()
            logger.warning("OTEL export failed; dropping %d span(s): %s", len(spans), exc)


# ── Core span builder ─────────────────────────────────────────────────────────
def _build_span(
    name: str,
    kind: str,
    trace_id: str,
    span_id: str,
    parent_span_id: Optional[str],
    start_ns: str,
    end_ns: str,
    attrs: list,
    status_code: int,
    status_msg: str = "",
) -> dict:
    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": kind,
        "startTimeUnixNano": start_ns,
        "endTimeUnixNano": end_ns,
        "attributes": attrs,
        "status": {"code": status_code, "message": status_msg},
    }
    if parent_span_id:
        span["parentSpanId"] = parent_span_id
    return span


# ── Continuity helpers ────────────────────────────────────────────────────────
def _common_attrs(session_id: str, **extra) -> list:
    trace_id = _ensure_trace_id(session_id)
    exec_id = _ensure_execution_id(session_id)
    attrs = [_attr(_GEN_AI_SESSION, session_id), _attr(_GEN_AI_EXECUTION, exec_id)]
    for k, v in extra.items():
        if v:
            attrs.append(_attr(k, v))
    return attrs


# ── Hook handlers ────────────────────────────────────────────────────────────

#: No-op stub — actual seeds are in conversation_loop / turn_finalizer.
on_session_start = lambda **kw: None


def on_session_reset(session_id: str, **kw) -> None:
    exporter_cfg = _exporter()
    if not exporter_cfg:
        return
    exporter = _Exporter(exporter_cfg)
    tid = _ensure_trace_id(session_id)
    now = _now_ns()
    extra = {k: v for k, v in kw.items() if v}
    attrs = _common_attrs(session_id, **extra)
    span = _build_span(
        "hermes.session", KIND_INTERNAL, tid, _span_id(), None, now, now, attrs, STATUS_UNSET
    )
    try:
        exporter.export([span])
    except Exception:
        pass


def on_session_finalize(session_id: str, reason: str = "", **kw) -> None:
    exporter_cfg = _exporter()
    if not exporter_cfg:
        return
    exporter = _Exporter(exporter_cfg)
    tid = _ensure_trace_id(session_id)
    now = _now_ns()
    extra = dict(kw)
    if reason:
        extra[_HERMES_FINALIZE] = reason
    attrs = _common_attrs(session_id, **extra)
    span = _build_span(
        "hermes.session", KIND_INTERNAL, tid, _span_id(), None, now, now, attrs, STATUS_OK
    )
    try:
        exporter.export([span])
    except Exception:
        pass


def on_session_end(session_id: str, **kw) -> None:
    """Intentionally quiet — on_session_finalize handles session-level close."""
    pass


def post_api_request(
    session_id: str,
    turn_id: str,
    api_request_id: str,
    provider: str,
    model: str,
    parent_session_id: str = "",
    resume_from: str = "",
    response_model: str = "",
    finish_reason: str = "",
    api_duration: float = 0.0,
    usage: Optional[dict] = None,
    **kw,
) -> None:
    exporter_cfg = _exporter()
    if not exporter_cfg:
        return
    exporter = _Exporter(exporter_cfg)
    tid = _ensure_trace_id(session_id)
    now = _now_ns()
    end_ns = str(int(time.time() * 1e9))
    span_name = f"gen_ai.chat {provider}"

    attrs = _common_attrs(session_id)
    attrs.append(_attr(_GEN_AI_OP, "chat"))
    attrs.append(_attr(_GEN_AI_SYSTEM, provider))
    attrs.append(_attr(_GEN_AI_REQ_MODEL, model))
    attrs.append(_attr(_GEN_AI_RESP_MODEL, response_model or model))
    if usage:
        attrs.append(_attr(_GEN_AI_USAGE_IN, str(usage.get("prompt_tokens", ""))))
        attrs.append(_attr(_GEN_AI_USAGE_OUT, str(usage.get("completion_tokens", ""))))
        attrs.append(_attr(_GEN_AI_USAGE_TOTAL, str(usage.get("total_tokens", ""))))
    if resume_from:
        attrs.append(_attr(_HERMES_RESUME, resume_from))
    cron_job = os.environ.get("HERMES_CRON_JOB_ID")
    if cron_job:
        attrs.append(_attr(_HERMES_CRON_JOB, cron_job))
    attrs.append(_attr(_HERMES_API_REQ_ID, api_request_id))

    status_code = STATUS_OK if str(finish_reason) != "error" else STATUS_ERROR
    span = _build_span(span_name, KIND_CLIENT, tid, _span_id(), None, now, end_ns, attrs, status_code)
    try:
        exporter.export([span])
    except Exception:
        pass


def api_request_error(
    session_id: str,
    provider: str,
    model: str,
    error: Optional[dict] = None,
    reason: str = "",
    retryable: bool = False,
    retry_count: int = 0,
    **kw,
) -> None:
    exporter_cfg = _exporter()
    if not exporter_cfg:
        return
    exporter = _Exporter(exporter_cfg)
    tid = _ensure_trace_id(session_id)
    now = _now_ns()
    err_dict = error or {}
    err_msg = err_dict.get("message", reason or "unknown")
    attrs = _common_attrs(session_id)
    attrs.append(_attr(_GEN_AI_OP, "chat"))
    attrs.append(_attr(_GEN_AI_SYSTEM, provider))
    attrs.append(_attr(_GEN_AI_REQ_MODEL, model))
    attrs.append(_attr(_ERROR_TYPE, err_dict.get("type", reason)))
    attrs.append(_attr(_HERMES_RETRYABLE, retryable))
    span = _build_span(
        f"gen_ai.chat {provider}", KIND_CLIENT, tid, _span_id(), None, now, now, attrs, STATUS_ERROR, err_msg
    )
    try:
        exporter.export([span])
    except Exception:
        pass


def post_tool_call(
    session_id: str,
    tool_call_id: str,
    tool_name: str,
    args: Optional[dict] = None,
    result: str = "",
    status: str = "ok",
    duration_ms: float = 0.0,
    **kw,
) -> None:
    exporter_cfg = _exporter()
    if not exporter_cfg:
        return
    exporter = _Exporter(exporter_cfg)
    tid = _ensure_trace_id(session_id)
    now = _now_ns()
    end_ns = str(int(time.time() * 1e9))
    args_json = json.dumps(args or {}) if args else "{}"
    span_name = f"hermes.tool {tool_name}"
    attrs = _common_attrs(session_id)
    attrs.append(_attr(_GEN_AI_TOOL_NAME, tool_name))
    attrs.append(_attr(_GEN_AI_TOOL_CALL_ID, tool_call_id))
    attrs.append(_attr(_HERMES_TOOL_STATUS, status))
    attrs.append(_attr(_HERMES_TOOL_DURATION, duration_ms))
    attrs.append(_attr(_HERMES_TOOL_ARGS_SIZE, len(args_json)))
    if result:
        attrs.append(_attr(_HERMES_TOOL_RESULT_SIZE, len(result)))
    status_code = STATUS_OK if str(status) == "ok" else STATUS_ERROR
    span = _build_span(span_name, KIND_INTERNAL, tid, _span_id(), None, now, end_ns, attrs, status_code)
    try:
        exporter.export([span])
    except Exception:
        pass