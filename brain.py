import httpx
import json
import logging
import config

logger = logging.getLogger(__name__)


async def stream_llm_response(conversation_history: list[dict], model: str = None):
    """
    Stream text tokens from OpenRouter.
    Yields content strings as they arrive via SSE.
    """
    selected_model = model if model else config.LLM_MODEL
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://voicebot.local",
        "X-Title": "AI Voice Bot",
    }
    payload = {
        "model": selected_model,
        "messages": conversation_history,

        "stream": True,
        "temperature": 0.7,
        "max_tokens": 200,  # Keep voice responses short
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    logger.error(f"OpenRouter API error {response.status_code}: {error_body.decode()}")
                    return

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            logger.debug(f"Skipping unparseable chunk: {e}")
                            continue
    except httpx.TimeoutException:
        logger.error("OpenRouter streaming request timed out")
    except Exception as e:
        logger.error(f"OpenRouter streaming error: {e}")
