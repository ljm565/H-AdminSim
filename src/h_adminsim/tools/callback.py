from langchain_core.outputs.llm_result import LLMResult
from langchain.callbacks.base import BaseCallbackHandler



class TokenUsageCallback(BaseCallbackHandler):
    def __init__(self):
        self.token_usage = {
            "prompt_tokens": [],
            "completion_tokens": [],
            "total_tokens": [],
            "reasoning_tokens": [],
        }


    def on_llm_end(self, response: LLMResult, **kwargs):
        """
        Callback handler that extracts token usage statistics after each LLM call.

        Args:
            response (LLMResult): The response object returned by the LLM, containing generation results and usage metadata.
        """
        for generations in response.generations:
            for generation in generations:
                usage = generation.message.usage_metadata
                if usage:
                    self.token_usage["prompt_tokens"].append(usage.get("input_tokens", 0))
                    self.token_usage["completion_tokens"].append(usage.get("output_tokens", 0))
                    self.token_usage["total_tokens"].append(usage.get("total_tokens", 0))
                    self.token_usage["reasoning_tokens"].append(usage.get("output_token_details", {}).get("reasoning", 0))