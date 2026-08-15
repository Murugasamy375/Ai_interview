import httpx
import json
import logging
from typing import Dict, Any, Union
from app import config

logger = logging.getLogger("app.services.llm")

async def call_grok(
    system_prompt: str,
    user_prompt: str,
    api_key: str = "",
    json_response: bool = True
) -> Union[Dict[str, Any], str]:
    """
    Call the xAI Grok API using direct HTTP requests.
    Supports dynamic API key injection and structured JSON output.
    """
    # 1. Resolve API Key
    effective_api_key = api_key or config.GROK_API_KEY
    if not effective_api_key:
        raise ValueError(
            "Grok API Key is missing. Please set the GROK_API_KEY environment variable "
            "or provide it in the 'X-Grok-API-Key' request header."
        )

    # 2. Determine Endpoint and Model
    # Groq keys usually start with 'gsk_'
    is_groq = effective_api_key.startswith("gsk_")
    if is_groq:
        url = "https://api.groq.com/openai/v1/chat/completions"
        # If model is not set or refers to a Grok model, use llama-3.3-70b-versatile
        if config.GROK_MODEL.startswith("grok"):
            model = "llama-3.3-70b-versatile"
        else:
            model = config.GROK_MODEL
        logger.info(f"Groq API Key detected. Using Groq endpoint with model: {model}")
    else:
        url = "https://api.x.ai/v1/chat/completions"
        model = config.GROK_MODEL
        logger.info(f"xAI Grok API Key detected. Using Grok endpoint with model: {model}")

    headers = {
        "Authorization": f"Bearer {effective_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1
    }

    if json_response:
        payload["response_format"] = {"type": "json_object"}
    
    logger.info(f"Calling API ({model}) at {url}...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            logger.error(f"API call failed: {response.status_code} - {response.text}")
            raise Exception(f"API call failed: {response.status_code} - {response.text}")
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        if json_response:
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Grok JSON response: {content}")
                raise Exception(f"Grok returned invalid JSON: {str(e)}")
        
        return content
