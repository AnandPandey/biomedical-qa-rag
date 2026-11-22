"""Shared LLM layer: cached client, prompt template, answer parser and generic runner."""
# This file marks the folder `bmqa/llm` as a Python package, so other code can write
# `from bmqa.llm.client import ...`. It contains no code on purpose: each module is imported
# directly by name (cache, client, prompts, parse, run), which keeps imports cheap and explicit.
