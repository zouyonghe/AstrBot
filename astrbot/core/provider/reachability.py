import asyncio
import os
from dataclasses import dataclass

from astrbot import logger
from astrbot.core.provider.entities import ProviderType
from astrbot.core.provider.provider import RerankProvider
from astrbot.core.utils.astrbot_path import get_astrbot_path


@dataclass
class ReachabilityResult:
    """Structured provider reachability result."""

    available: bool
    error_code: str | None = None
    error: str | None = None
    response_snippet: str | None = None


def _extract_err_code(exc: Exception) -> str:
    return (
        getattr(exc, "status_code", None)
        or getattr(exc, "code", None)
        or getattr(exc, "error_code", None)
        or exc.__class__.__name__
    )


def _extract_response_snippet(response) -> str | None:
    """Best-effort snippet extraction for chat responses."""
    try:
        if hasattr(response, "completion_text") and response.completion_text:
            return (
                response.completion_text[:70] + "..."
                if len(response.completion_text) > 70
                else response.completion_text
            )
        if hasattr(response, "result_chain") and response.result_chain:
            text = response.result_chain.get_plain_text()
            return text[:70] + "..." if len(text) > 70 else text
    except Exception:
        return None
    return None


def _cleanup_temp_file(path: str):
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError as exc:
            logger.debug("Failed to cleanup temp file %s: %s", path, exc)


async def check_provider_reachability(
    provider,
    *,
    timeout: float = 45.0,
    stt_sample_optional: bool = False,
    cleanup_tts_file: bool = True,
) -> ReachabilityResult:
    """
    Run a reachability/health check against a provider.

    Returns a structured result with availability and optional error info.
    """
    meta = provider.meta()
    provider_capability_type = meta.provider_type
    provider_name = provider.provider_config.get("id", "Unknown Provider")

    logger.debug(
        "Checking provider reachability: name=%s id=%s type=%s model=%s",
        provider_name,
        getattr(meta, "id", "Unknown ID"),
        getattr(provider_capability_type, "value", "unknown"),
        getattr(meta, "model", "Unknown Model"),
    )

    try:
        if provider_capability_type == ProviderType.CHAT_COMPLETION:
            response = await asyncio.wait_for(
                provider.text_chat(prompt="REPLY `PONG` ONLY"),
                timeout=timeout,
            )
            if response is not None:
                snippet = _extract_response_snippet(response)
                logger.info(
                    "Provider %s (ID: %s) is available. Response snippet: %s",
                    provider_name,
                    getattr(meta, "id", "Unknown ID"),
                    snippet,
                )
                return ReachabilityResult(True, response_snippet=snippet)
            logger.warning(
                "Provider %s (ID: %s) test call returned None.",
                provider_name,
                getattr(meta, "id", "Unknown ID"),
            )
            return ReachabilityResult(
                False,
                "EMPTY_RESPONSE",
                "Test call returned None, but expected an LLMResponse object.",
            )

        if provider_capability_type == ProviderType.EMBEDDING:
            embedding_result = await asyncio.wait_for(
                provider.get_embedding("health_check"),
                timeout=timeout,
            )
            if isinstance(embedding_result, list) and (
                not embedding_result or isinstance(embedding_result[0], float)
            ):
                return ReachabilityResult(True)
            logger.warning(
                "Embedding test failed for %s: unexpected result type %s",
                provider_name,
                type(embedding_result),
            )
            return ReachabilityResult(
                False,
                "INVALID_EMBEDDING",
                f"Embedding test failed: unexpected result type {type(embedding_result)}",
            )

        if provider_capability_type == ProviderType.TEXT_TO_SPEECH:
            audio_result = await asyncio.wait_for(
                provider.get_audio("你好"),
                timeout=timeout,
            )
            if isinstance(audio_result, str) and audio_result:
                if cleanup_tts_file:
                    _cleanup_temp_file(audio_result)
                return ReachabilityResult(True)
            logger.warning(
                "TTS test failed for %s: unexpected result type %s",
                provider_name,
                type(audio_result),
            )
            return ReachabilityResult(
                False,
                "INVALID_AUDIO",
                f"TTS test failed: unexpected result type {type(audio_result)}",
            )

        if provider_capability_type == ProviderType.SPEECH_TO_TEXT:
            sample_audio_path = os.path.join(
                get_astrbot_path(),
                "samples",
                "stt_health_check.wav",
            )
            if not os.path.exists(sample_audio_path):
        if stt_sample_optional:
            if hasattr(provider, "get_text"):
                return ReachabilityResult(True)
            return ReachabilityResult(
                False,
                "STT_SAMPLE_MISSING",
                "STT test skipped: provider missing get_text and sample file not found.",
            )
        logger.warning(
            "STT test for %s failed: sample audio file not found at %s",
            provider_name,
            sample_audio_path,
        )
                return ReachabilityResult(
                    False,
                    "STT_SAMPLE_MISSING",
                    "STT test failed: sample audio file not found.",
                )

            text_result = await asyncio.wait_for(
                provider.get_text(sample_audio_path),
                timeout=timeout,
            )
            if isinstance(text_result, str) and text_result:
                snippet = text_result[:70] + "..." if len(text_result) > 70 else text_result
                logger.info(
                    "Provider %s (ID: %s) is available. Response snippet: %s",
                    provider_name,
                    getattr(meta, "id", "Unknown ID"),
                    snippet,
                )
                return ReachabilityResult(True, response_snippet=snippet)
            logger.warning(
                "STT test for %s failed: unexpected result type %s",
                provider_name,
                type(text_result),
            )
            return ReachabilityResult(
                False,
                "INVALID_TEXT",
                f"STT test failed: unexpected result type {type(text_result)}",
            )

        if provider_capability_type == ProviderType.RERANK:
            if isinstance(provider, RerankProvider):
                await asyncio.wait_for(
                    provider.rerank("Apple", documents=["apple", "banana"]),
                    timeout=timeout,
                )
                return ReachabilityResult(True)
            return ReachabilityResult(
                False,
                "NOT_RERANK_PROVIDER",
                "Provider is not RerankProvider",
            )

        # Unknown types fall back to get_models if available
        if hasattr(provider, "get_models"):
            await asyncio.wait_for(provider.get_models(), timeout=timeout)
        return ReachabilityResult(
            True,
            error="This provider type is not tested and is assumed to be available.",
            response_snippet="This provider type is not tested and is assumed to be available.",
        )

    except asyncio.TimeoutError:
        return ReachabilityResult(False, "TIMEOUT", "Connection timed out during test call.")
    except Exception as exc:
        err_code = _extract_err_code(exc)
        err_reason = str(exc)
        logger.warning(
            "Provider reachability check failed: id=%s type=%s code=%s reason=%s",
            getattr(meta, "id", "Unknown ID"),
            getattr(provider_capability_type, "name", "unknown"),
            err_code,
            err_reason,
        )
        return ReachabilityResult(False, err_code, err_reason)


def build_provider_status_info(provider, reachability: ReachabilityResult) -> dict:
    """Build dashboard-friendly status dict from reachability result."""
    meta = provider.meta()
    provider_capability_type = meta.provider_type
    provider_name = provider.provider_config.get("id", "Unknown Provider")

    status = "available" if reachability.available else "unavailable"
    status_info = {
        "id": getattr(meta, "id", "Unknown ID"),
        "model": getattr(meta, "model", "Unknown Model"),
        "type": provider_capability_type.value if provider_capability_type else "unknown",
        "name": provider_name,
        "status": status,
        "error": reachability.error,
    }

    return status_info
