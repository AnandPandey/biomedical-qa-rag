"""LLM clients implementing the `LLMClient` protocol from bmqa.schema.

Only the provider class depends on a vendor SDK; it is imported lazily, so the package (and the
tests, which use FakeLLMClient) work without the SDK installed.
"""

# Allows modern type hints such as `str | None` without evaluating them at runtime.
from __future__ import annotations

# datetime gives the current time; UTC makes timestamps time-zone independent.
from datetime import UTC, datetime

# Path represents file/folder paths (used for the cache folder argument).
from pathlib import Path

# Our on-disk cache: the default folder, the cache class and the key-building function.
from bmqa.llm.cache import DEFAULT_CACHE_DIR, ResponseCache, cache_key

# LABELS = ("yes", "no", "maybe"); LLMResponse = the shared result type every client returns
# (text, model, input_tokens, output_tokens, cached).
from bmqa.schema import LABELS, LLMResponse

# Structured-output schema: the reply must be JSON with exactly these keys.
# This is a JSON Schema sent to the API, which then forces the model to reply in this shape.
ANSWER_SCHEMA = {
    "type": "object",  # the reply is one JSON object { ... }
    "properties": {
        # "answer" must be a string and one of the allowed labels: yes / no / maybe.
        "answer": {"type": "string", "enum": list(LABELS)},
        # "rationale" is free text: a short explanation of the answer.
        "rationale": {"type": "string"},
        # "citations" is a list of whole numbers: the numbers [1], [2]... of cited documents.
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    # All three keys must be present in every reply.
    "required": ["answer", "rationale", "citations"],
    # No other keys are allowed, so the reply is always easy to parse.
    "additionalProperties": False,
}

# The fixed reply the fake client returns by default: a valid "yes" answer with no citations.
FAKE_REPLY = '{"answer": "yes", "rationale": "Fake reply for tests.", "citations": []}'


# A pretend LLM used in tests and dry runs. It returns a canned reply instantly, needs no API
# key, makes no network calls and therefore costs nothing.
class FakeLLMClient:
    """Deterministic client for tests and --dry-run. Never touches the network."""

    # Inputs: `reply` is either a fixed string or a function that turns a prompt into a reply;
    # `model` is the name recorded in predictions (default "fake-llm").
    def __init__(self, reply=FAKE_REPLY, model: str = "fake-llm"):
        # Model name, read by the runner and stored in every Prediction.
        self.model = model
        self._reply = reply  # a string, or a function prompt -> string
        # Every prompt received is remembered, so tests can inspect what was sent.
        self.prompts: list[str] = []
        # Why the last reply ended (same attribute the real client has); None before any call.
        self.last_stop_reason: str | None = None

    # Produce a reply for one prompt. Input: prompt text and max_tokens (ignored here, but
    # required so the fake has the same interface as the real client). Output: LLMResponse.
    def generate(self, prompt: str, *, max_tokens: int) -> LLMResponse:
        # Record the prompt for later inspection by tests.
        self.prompts.append(prompt)
        # If the reply is a function, call it with the prompt; otherwise use the fixed string.
        text = self._reply(prompt) if callable(self._reply) else self._reply
        # Pretend the model finished normally ("end_turn" = a complete answer).
        self.last_stop_reason = "end_turn"
        # Package the reply in the shared response type.
        return LLMResponse(
            text=text,  # the reply text
            model=self.model,
            input_tokens=len(prompt.split()),  # rough fake token count: number of words
            output_tokens=len(text.split()),  # same rough count for the reply
            cached=False,  # the fake never uses the cache
        )


# The real client: talks to Claude through the official `anthropic` Python SDK, asks for
# structured JSON output, and stores every reply in the on-disk cache.
class AnthropicClient:
    """Claude via the official Anthropic SDK, with structured JSON output and a disk cache.

    The SDK reads the API key from the ANTHROPIC_API_KEY environment variable.
    """

    # Inputs: model name, effort level ("low", ...), run number (part of the cache key) and
    # the cache folder. The API key is NOT passed in: the SDK reads it from the environment,
    # so it never appears in code, configs or logs. Never print it: anyone who sees it can
    # spend money on the team's account.
    def __init__(
        self, model: str, effort: str, run: int, cache_dir: str | Path = DEFAULT_CACHE_DIR
    ):
        # Imported here ("lazily") instead of at the top of the file, so the rest of the
        # package and the tests still work on machines where the SDK is not installed.
        import anthropic  # needs the `llm` extra

        # Remember the settings; they are used for every request and in the cache key.
        self.model = model
        self.effort = effort
        self.run = run
        # Open (or create on first write) the disk cache in the given folder.
        self.cache = ResponseCache(cache_dir)
        # Why the last reply stopped (e.g. "end_turn", "max_tokens"); read by the runner.
        self.last_stop_reason: str | None = None
        # The SDK retries rate-limit (429) and server errors itself; other errors stop the run.
        # max_retries=5 means up to 5 automatic retries with waiting in between.
        self._sdk = anthropic.Anthropic(max_retries=5)

    # Get a reply for one prompt, from the cache if possible, otherwise from the API.
    # Input: prompt text and max_tokens (longest allowed reply). Output: LLMResponse.
    def generate(self, prompt: str, *, max_tokens: int) -> LLMResponse:
        # Fingerprint of this exact request (model, effort, length limit, prompt and run).
        key = cache_key(
            model=self.model, effort=self.effort, max_tokens=max_tokens, prompt=prompt, run=self.run
        )
        # Try the cache first; this is None if the request was never made before.
        record = self.cache.get(key)
        # Remember whether this answer came from the cache (cached answers cost nothing).
        cached = record is not None
        # Cache miss: we must actually call the API (this costs money).
        if record is None:
            # No temperature/top_p/top_k: the current API version rejects them.
            # Send one user message and ask for JSON that matches ANSWER_SCHEMA.
            response = self._sdk.messages.create(
                model=self.model,  # which Claude model to use
                max_tokens=max_tokens,  # stop the reply after this many tokens
                # output_config sets the reasoning effort and the required JSON reply format.
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
                },
                # The conversation: a single message from the user containing the prompt.
                messages=[{"role": "user", "content": prompt}],
            )
            # Keep only what we need from the response, as a plain dict that can be cached.
            record = {
                # The reply is a list of content blocks; take the text of the first "text"
                # block, or "" if there is none (next() with a default avoids an error).
                "text": next((b.text for b in response.content if b.type == "text"), ""),
                "model": self.model,
                # Token counts reported by the API; used later for the cost estimate.
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                # Why the model stopped, e.g. "end_turn" (finished) or "max_tokens" (cut off).
                "stop_reason": response.stop_reason,
                # When the call was made, as an ISO text timestamp in UTC (to the second).
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            # Save the record so the same request in the same run is free next time.
            self.cache.put(key, record)
        # Expose the stop reason (fresh or cached) so the runner can spot cut-off answers.
        self.last_stop_reason = record["stop_reason"]
        # Return the reply in the shared response type, flagged as cached or not.
        return LLMResponse(
            text=record["text"],
            model=self.model,
            input_tokens=record["input_tokens"],
            output_tokens=record["output_tokens"],
            cached=cached,
        )


# The open-weights client: runs Qwen2.5-7B-Instruct via transformers and PyTorch with 4-bit
# quantization (BitsAndBytesConfig). Designed for Google Colab T4 GPU or local GPU inference.
class QwenClient:
    """Qwen2.5-Instruct via transformers and PyTorch, with 4-bit quantization and a disk cache."""

    # Inputs: model name, run number, cache folder, 4-bit quantization flag, and device mapping.
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        run: int = 1,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ):
        # Lazy imports so environments without torch/transformers/bitsandbytes don't crash.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model
        self.run = run
        self.cache = ResponseCache(cache_dir)
        self.last_stop_reason: str | None = None

        # Load the Hugging Face tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        # Configure 4-bit NormalFloat4 quantization if requested and CUDA is available.
        kwargs = {}
        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = device_map

        # Load the model weights.
        if torch.cuda.is_available():
            self._model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                **kwargs,
            )
        else:
            # Fallback for CPU / mocked testing environments.
            self._model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float32,
            )

    # Produce a reply for one prompt, using cache if present, otherwise running inference.
    def generate(self, prompt: str, *, max_tokens: int) -> LLMResponse:
        # Cache key fingerprints the request.
        key = cache_key(
            model=self.model,
            effort="none",
            max_tokens=max_tokens,
            prompt=prompt,
            run=self.run,
        )
        record = self.cache.get(key)
        cached = record is not None

        if record is None:
            import torch

            # Apply official chat template if available.
            if getattr(self.tokenizer, "chat_template", None):
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt

            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            device = getattr(self._model, "device", "cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_tokens = int(inputs["input_ids"].shape[1])

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # greedy decoding for reproducible evaluation
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Slicing past the input token length yields only the newly generated tokens.
            generated_ids = outputs[0][input_tokens:]
            output_tokens = int(len(generated_ids))
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Record stop reason based on whether output hit token limit.
            stop_reason = "max_tokens" if output_tokens >= max_tokens else "end_turn"

            record = {
                "text": text,
                "model": self.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "stop_reason": stop_reason,
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            self.cache.put(key, record)

        self.last_stop_reason = record["stop_reason"]
        return LLMResponse(
            text=record["text"],
            model=self.model,
            input_tokens=record["input_tokens"],
            output_tokens=record["output_tokens"],
            cached=cached,
        )


# Factory function: build the right client from the config.
# Inputs: `llm_cfg` = the `llm:` block of a YAML config (provider, model, effort, max_tokens);
# `run` = run number; `dry_run` = True to use the free fake client; `cache_dir` = cache folder.
# Output: a FakeLLMClient, AnthropicClient, or QwenClient (all implement LLMClient protocol).
def make_client(
    llm_cfg: dict, run: int, dry_run: bool = False, cache_dir: str | Path = DEFAULT_CACHE_DIR
):
    """Create the client described by the `llm` block of an experiment config."""
    # Dry run: return the fake client, so no API key or GPU is needed and nothing is spent.
    if dry_run:
        return FakeLLMClient()
    # Which LLM company/service the config asks for.
    provider = llm_cfg["provider"]
    # Anthropic (Claude): create the real client with the configured model and effort.
    if provider == "anthropic":
        return AnthropicClient(llm_cfg["model"], llm_cfg["effort"], run, cache_dir)
    # Qwen (open-weights): create the local/Colab client with 4-bit quantization.
    if provider == "qwen":
        return QwenClient(
            model=llm_cfg.get("model", "Qwen/Qwen2.5-7B-Instruct"),
            run=run,
            cache_dir=cache_dir,
            load_in_4bit=llm_cfg.get("load_in_4bit", True),
        )
    # Any other provider name is a mistake in the config: fail loudly with a clear message.
    raise ValueError(f"unknown LLM provider {provider!r}")
