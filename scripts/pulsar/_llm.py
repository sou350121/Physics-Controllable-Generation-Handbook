"""Shared LLM transport for the Pulsar physics-gen pipeline.

ONE helper for every LLM call in this repo. Before 2026-09-10 there were two
hand-rolled `call_qwen` copies (rate.py, run_weekly.py) with incompatible
signatures and slightly different retry logic; both are now thin wrappers that
preserve their original signature and delegate here.

Provider order: DeepSeek `deepseek-flash` primary, DashScope qwen fallback.

WHY THE SWITCH
The repo's DASHSCOPE_API_KEY Actions secret (last rotated 2026-06-13) was
revoked. Every rating call in CI returned `HTTP Error 401: Unauthorized`, and
because rate.py catches per-paper exceptions and stamps 📖, the workflow kept
reporting success while writing placeholder sheets. reports/physics-gen-daily/
from 2026-08-31 to 2026-09-04 are all "⚡0 · 🔧0 · 📖80"; the last genuinely
rated daily is 2026-08-28.

MEASURED FACTS — do not "clean these up" without re-measuring
  * Endpoint https://api.deepseek.com/chat/completions, OpenAI-compatible.
    Model id `deepseek-flash`. (`deepseek-v4-flash`, `deepseek-chat` and
    `deepseek-reasoner` are aliases that resolve to it; `deepseek-v4.1-*` does
    not exist. The only other real model is `deepseek-v4-pro`.)
  * NEVER send response_format={"type":"json_object"} to DeepSeek. On a ~50-item
    rating prompt, bare was 5/5 valid JSON and json_object was 1/5 — DeepSeek
    emits a literal {"type": "json_object"} line ahead of the real payload.
    json_schema strict mode answers "unavailable". Re-measured 2026-09-10 on
    this repo's own single-paper prompt: json_object roughly doubled reasoning
    tokens and latency (5.4-10.2s vs 4.5-6.4s bare) and degraded the verdict
    (🔧 where bare said ❌ three times out of three). There is no upside.
  * DeepSeek's max_tokens controls HOW MUCH IT REASONS, and reasoning counts
    against the same budget. Measured on this repo's single-paper prompt:
    max_tokens=1024 -> finish_reason="length", 1024 reasoning tokens and
    EMPTY content; 4096 -> 333 reasoning, valid; 8192 -> 1522, valid;
    65536 -> 498 reasoning, valid, 3.4s and the best-grounded reason text.
    A single-item prompt self-limits rather than expanding to fill 65536, so
    the production-proven 65536 is both the safest and a fast choice here.
    The ONLY way to switch reasoning off is {"thinking": {"type": "disabled"}}
    — `reasoning_effort` has no effect.
  * DeepSeek can answer HTTP 200 with a body json.loads() refuses. That is a
    FAILURE that must reach the fallback, never a placeholder. See salvage_json.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from _config import (
    DASHSCOPE_BASE_URL, LLM_MODEL, LLM_TIMEOUT, LLM_RETRY, LLM_RETRY_BACKOFF,
    DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, DEEPSEEK_MAX_TOKENS,
)

# Retryable transport-level statuses (rate limit + upstream hiccups).
_RETRY_CODES = (429, 500, 502, 503)

# Circuit breaker. rate.py calls this module once PER PAPER (~99 papers/day), so
# a dead provider would otherwise burn LLM_RETRY attempts * backoff on every
# single paper and blow the workflow's 30-minute timeout. After this many
# consecutive hard failures we stop dialling that provider for the rest of the
# process and say so once.
_BREAKER_THRESHOLD = 3
_state = {"deepseek_fails": 0, "qwen_fails": 0, "deepseek_open": False, "qwen_open": False}

# Which provider actually answered last — post.py/run_weekly.py stamp this into
# the report header so a sheet always states how it was really produced.
LAST_PROVIDER = ""


def _read_env_file_key(name: str) -> str:
    """Fallback key lookup in ~/.clawdbot/.env (cron runs carry no env vars).

    CI supplies the key through the workflow `env:` block; this only matters for
    local/cron execution. Never hardcode a key here.
    """
    path = Path.home() / ".clawdbot" / ".env"
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip("'\"")
    except Exception:
        pass
    return ""


def get_key(name: str) -> str:
    """Resolve an API key: process env first, then ~/.clawdbot/.env."""
    return (os.environ.get(name, "") or "").strip() or _read_env_file_key(name)


def _post_chat(url: str, api_key: str, payload: dict, timeout: int) -> dict:
    """POST an OpenAI-compatible chat payload. Returns {ok, content} or {ok, error}.

    Treats three shapes as failure that the caller must fall back on, rather
    than as content:
      - HTTP error (401 revoked key, 4xx, exhausted 5xx retries)
      - empty content (a reasoning model can spend the whole budget thinking and
        still return finish_reason="stop" with nothing to show for it)
      - finish_reason=="length" (truncated JSON parses as garbage and would
        silently downgrade every paper to 📖)
    """
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(LLM_RETRY):
        # Rebuild the Request each attempt — a consumed Request cannot be replayed.
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                obj = json.loads(r.read().decode("utf-8", errors="replace"))
            choice = (obj.get("choices") or [{}])[0] or {}
            content = (choice.get("message") or {}).get("content") or ""
            if not content.strip():
                return {"ok": False, "error": "empty_content"}
            if choice.get("finish_reason") == "length":
                return {"ok": False, "error": "truncated_output"}
            return {"ok": True, "content": content}
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            if e.code in _RETRY_CODES and attempt < LLM_RETRY - 1:
                time.sleep(LLM_RETRY_BACKOFF * (attempt + 1))
                continue
            return {"ok": False, "error": f"http_{e.code}", "detail": detail}
        except Exception as e:  # URLError, timeout, malformed envelope
            if attempt < LLM_RETRY - 1:
                time.sleep(LLM_RETRY_BACKOFF * (attempt + 1))
                continue
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    return {"ok": False, "error": "max_retries"}


def salvage_json(content: str):
    """Best-effort recovery of a JSON object/array from a model response.

    Returns the parsed value, or None if nothing usable could be recovered.
    Adapted from the VLA pipeline's _salvage_rating_array() — same two layers,
    but this repo rates ONE paper per call and so expects a JSON *object*, where
    VLA batches ~50 papers into one array.

    Layer 1 — prose prefix / code fence: the model narrates before emitting the
      payload. Strip fences, then raw_decode from the first '{' (or '[').
    Layer 2 — one malformed element inside an array: typically an unescaped '"'
      in a Chinese `reason`. Split on element boundaries and parse each element
      alone, dropping only the broken one instead of the whole sheet. (On the
      VLA side this recovered 49 of 51 papers on a day that had produced zero.)
    """
    if not content:
        return None
    s = content.strip()
    if s.startswith("```"):
        s = "\n".join(l for l in s.splitlines()
                      if not l.strip().startswith("```")).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    # Layer 1: decode from the first opening bracket of either kind.
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    try:
        value, _ = json.JSONDecoder().raw_decode(s[start:])
        return value
    except Exception:
        pass

    # Layer 2: array with one broken element — salvage the rest.
    i = s.find("[")
    if i < 0:
        return None
    j = s.rfind("]")
    body = s[i + 1:j] if j > i else s[i + 1:]
    out = []
    for chunk in re.split(r"\}\s*,\s*\{", body):
        chunk = chunk.strip()
        if not chunk:
            continue
        if not chunk.startswith("{"):
            chunk = "{" + chunk
        if not chunk.endswith("}"):
            chunk = chunk + "}"
        try:
            parsed = json.loads(chunk)
            if isinstance(parsed, dict):
                out.append(parsed)
        except Exception:
            continue
    if out:
        print(f"  NOTE: salvaged {len(out)} element(s) from a malformed array",
              file=sys.stderr)
    return out or None


def _check_body(res: dict, expect_json: bool, validate) -> dict:
    """Reject an HTTP 200 whose body is not actually usable.

    Two ways a 200 can be worthless:
      - salvage_json recovers nothing at all;
      - it recovers *something* that is not what the caller asked for. The
        canonical example is the {"type": "json_object"} marker DeepSeek leaks
        when response_format is set — valid JSON, zero information. We never send
        response_format, but a payload carrying no usable verdict is semantically
        unparseable either way, and must fall back rather than burn the paper.
    """
    if not (res.get("ok") and expect_json):
        return res
    parsed = salvage_json(res.get("content"))
    if parsed is None:
        return {"ok": False, "error": "unparseable_body"}
    if validate is not None and not validate(parsed):
        return {"ok": False, "error": "payload_failed_validation"}
    return res


def _try_deepseek(messages: list[dict], temperature: float, timeout: int,
                  expect_json: bool, validate) -> dict:
    key = get_key("DEEPSEEK_API_KEY")
    if not key:
        return {"ok": False, "error": "no_DEEPSEEK_API_KEY"}
    if _state["deepseek_open"]:
        return {"ok": False, "error": "circuit_open"}
    res = _post_chat(DEEPSEEK_BASE_URL + "/chat/completions", key, {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        # Load-bearing, see module docstring. NO response_format here, ever.
        "max_tokens": DEEPSEEK_MAX_TOKENS,
    }, timeout)
    return _check_body(res, expect_json, validate)


def _try_qwen(messages: list[dict], temperature: float, timeout: int,
              expect_json: bool, validate) -> dict:
    key = get_key("DASHSCOPE_API_KEY")
    if not key:
        return {"ok": False, "error": "no_DASHSCOPE_API_KEY"}
    if _state["qwen_open"]:
        return {"ok": False, "error": "circuit_open"}
    res = _post_chat(DASHSCOPE_BASE_URL + "/chat/completions", key, {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 8192,
        "enable_thinking": False,  # fallback path: stay inside the time budget
    }, timeout)
    return _check_body(res, expect_json, validate)


def _trip(provider: str, error: str) -> None:
    """Count a hard failure and open the breaker once it repeats."""
    kf, ko = f"{provider}_fails", f"{provider}_open"
    if error == "circuit_open":
        return
    _state[kf] += 1
    if _state[kf] >= _BREAKER_THRESHOLD and not _state[ko]:
        _state[ko] = True
        print(f"  BREAKER: {provider} failed {_state[kf]}x consecutively "
              f"(last: {error}) — no further {provider} calls this run",
              file=sys.stderr)


def chat(messages: list[dict], *, temperature: float = 0.1,
         timeout: int = LLM_TIMEOUT, expect_json: bool = True,
         validate=None) -> str:
    """DeepSeek first, qwen fallback. Returns assistant text.

    validate: optional predicate run on the salvaged JSON payload. Returning
    False marks the response unusable and moves on to the next provider, so a
    well-formed but content-free 200 degrades instead of poisoning a report.

    Raises RuntimeError when BOTH providers fail — callers must let that
    propagate into a visibly failed run rather than writing a placeholder.
    """
    global LAST_PROVIDER
    errors = []

    res = _try_deepseek(messages, temperature, timeout, expect_json, validate)
    if res.get("ok"):
        _state["deepseek_fails"] = 0
        LAST_PROVIDER = DEEPSEEK_MODEL
        return res["content"]
    err = res.get("error", "?")
    errors.append(f"deepseek({DEEPSEEK_MODEL}): {err}{_detail(res)}")
    _trip("deepseek", err)

    # Fallback. In CI this will itself fail while no live DashScope secret
    # exists — that is intended to be loud, not papered over.
    res = _try_qwen(messages, temperature, min(timeout, 240), expect_json, validate)
    if res.get("ok"):
        _state["qwen_fails"] = 0
        LAST_PROVIDER = LLM_MODEL
        return res["content"]
    err = res.get("error", "?")
    errors.append(f"qwen({LLM_MODEL}): {err}{_detail(res)}")
    _trip("qwen", err)

    raise RuntimeError("all LLM providers failed — " + " | ".join(errors))


def _detail(res: dict) -> str:
    d = (res.get("detail") or "").strip().replace("\n", " ")
    return f" [{d[:120]}]" if d else ""


def provider_banner() -> str:
    """Human-readable 'who actually rated this' string for report headers."""
    return LAST_PROVIDER or "unknown"


def preflight() -> None:
    """Log which providers have a key before doing any work."""
    ds = "yes" if get_key("DEEPSEEK_API_KEY") else "NO"
    qw = "yes" if get_key("DASHSCOPE_API_KEY") else "NO"
    print(f"  LLM providers: deepseek={DEEPSEEK_MODEL} key={ds} (primary) · "
          f"qwen={LLM_MODEL} key={qw} (fallback)", file=sys.stderr)
    if ds == "NO" and qw == "NO":
        print("  WARN: no LLM key available at all — this run will fail loudly.",
              file=sys.stderr)
