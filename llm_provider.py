"""
LLM Provider factory for NVIDIA NIM (OpenAI-compatible).
"""
import os
from dotenv import load_dotenv

load_dotenv(override=False)

from langchain_openai import ChatOpenAI


def get_llm(model: str = None, temperature: float = 1) -> ChatOpenAI:
    """
    Initialize and return a ChatOpenAI instance configured for NVIDIA NIM.
    
    Args:
        model: Model ID (defaults to LLM_MODEL env var or google/gemma-4-31b-it)
        temperature: Sampling temperature
        
    Returns:
        ChatOpenAI instance
        
    Raises:
        ValueError: If NVIDIA_API_KEY is missing or empty
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY environment variable is required. Get one at build.nvidia.com"
        )
    
    model_name = model or os.getenv("LLM_MODEL", "moonshotai/kimi-k2.5")
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
    return llm
