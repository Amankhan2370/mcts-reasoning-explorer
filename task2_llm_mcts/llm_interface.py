"""
LLM Interface - Generic wrapper for various LLM providers
API keys must be set via environment variables for security
"""

import os
from typing import Optional


class LLMInterface:
    """
    Generic LLM API wrapper.

    Supports: OpenAI, Anthropic, HuggingFace, OpenRouter

    Setup:
        export OPENAI_API_KEY="your-key"
        export ANTHROPIC_API_KEY="your-key"
        export HF_API_KEY="your-key"
        export OPENROUTER_API_KEY="your-key"
    """

    def __init__(self, provider: str = "openai", model: str = None, api_key: str = None):
        """
        Initialize LLM interface.

        Args:
            provider: "openai", "anthropic", "huggingface", or "openrouter"
            model: Model name (auto-selects if None)
            api_key: API key (reads from env if None) - NOT RECOMMENDED, use env vars
        """
        self.provider = provider.lower()

        # Read API key from environment variable (SECURE METHOD)
        if api_key is None:
            env_var = f"{provider.upper()}_API_KEY"
            self.api_key = os.getenv(env_var)
        else:
            self.api_key = api_key

        if not self.api_key:
            raise ValueError(
                f"API key not found. Please set {provider.upper()}_API_KEY environment variable.\n"
                f"Example: export {provider.upper()}_API_KEY='your-key-here'"
            )

        # Auto-select model based on provider
        if model is None:
            model_defaults = {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-3-5-haiku-20241022",
                "huggingface": "google/gemma-2-2b-it",
                "openrouter": "google/gemini-flash-1.5"
            }
            self.model = model_defaults.get(self.provider, "gpt-4o-mini")
        else:
            self.model = model

        # Initialize appropriate client
        self._initialize_client()
        self.call_count = 0

    def _initialize_client(self):
        """Initialize the appropriate API client based on provider."""
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)

        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)

        elif self.provider == "huggingface":
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.api_key)

        elif self.provider == "openrouter":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        Generate text from LLM.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum response length

        Returns:
            Generated text
        """
        self.call_count += 1

        try:
            if self.provider in ["openai", "openrouter"]:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.content[0].text.strip()

            elif self.provider == "huggingface":
                response = self.client.text_generation(
                    prompt,
                    model=self.model,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                return response.strip()

        except Exception as e:
            print(f"LLM API Error ({self.provider}): {e}")
            return ""

    def generate_reasoning_step(self, problem: str, current_reasoning: str = "") -> str:
        """Generate next reasoning step for a problem."""
        if current_reasoning:
            prompt = f"""Problem: {problem}

Reasoning so far:
{current_reasoning}

Continue the reasoning with the next logical step. Be concise and clear. Only provide the next step, not the full solution."""
        else:
            prompt = f"""Problem: {problem}

Provide the first step in solving this problem. Be concise and clear. Only provide the first step, not the full solution."""

        return self.generate(prompt, temperature=0.8, max_tokens=150)

    def complete_solution(self, problem: str, reasoning_chain: str) -> str:
        """Complete a solution given partial reasoning."""
        prompt = f"""Problem: {problem}

Reasoning:
{reasoning_chain}

Based on this reasoning, provide the final answer. Just give the answer, nothing else."""

        return self.generate(prompt, temperature=0.3, max_tokens=100)

    def get_stats(self):
        """Get usage statistics."""
        return {
            'calls': self.call_count,
            'provider': self.provider,
            'model': self.model
        }


def test_llm_interface():
    """
    Test the LLM interface.

    NOTE: Requires environment variable to be set first:
        export OPENAI_API_KEY="your-key"
        export ANTHROPIC_API_KEY="your-key"
        export HF_API_KEY="your-key"
        export OPENROUTER_API_KEY="your-key"
    """
    print("Testing LLM Interface...")
    print("NOTE: Make sure your API key environment variable is set!")

    try:
        # Try with provider from environment
        provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        llm = LLMInterface(provider=provider)

        response = llm.generate("What is 2+2? Answer in one word.")
        print(f"Simple test: {response}")
        print(f"Stats: {llm.get_stats()}")

    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTo use this interface, set an API key environment variable:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  export HF_API_KEY='your-key'")
        print("  export OPENROUTER_API_KEY='your-key'")


if __name__ == "__main__":
    test_llm_interface()
