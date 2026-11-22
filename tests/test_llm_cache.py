# Tests for the on-disk response cache (bmqa/llm/cache.py). Run them with `pytest`.
# Import the cache class and the function that builds cache keys.
from bmqa.llm.cache import ResponseCache, cache_key

# Shared request settings (everything except the run number), reused by the tests below.
ARGS = {"model": "m", "effort": "low", "max_tokens": 10, "prompt": "p"}


# Checks the key rules: same inputs -> same key; a different run or prompt -> different key.
def test_same_inputs_same_key_and_run_changes_key():
    # Identical inputs must always produce the identical key (otherwise the cache never hits).
    assert cache_key(**ARGS, run=1) == cache_key(**ARGS, run=1)
    # A different run number must give a different key, so runs 1 and 2 are independent calls.
    assert cache_key(**ARGS, run=1) != cache_key(**ARGS, run=2)
    # A different prompt ("q" instead of "p") must give a different key, so we never reuse
    # a reply that was produced for another prompt.
    assert cache_key(**ARGS, run=1) != cache_key(**{**ARGS, "prompt": "q"}, run=1)


# Checks that a stored record can be read back unchanged. `tmp_path` is a pytest fixture: a
# fresh empty temporary folder, so the test never touches the real cache.
def test_put_and_get(tmp_path):
    # A cache stored in the temporary folder.
    cache = ResponseCache(tmp_path)
    # A key for some request.
    key = cache_key(**ARGS, run=1)
    # Nothing stored yet, so get() must report "not cached" (None).
    assert cache.get(key) is None
    # Store a record; the accented character checks that non-ASCII text survives saving.
    cache.put(key, {"text": "hello é", "stop_reason": "end_turn"})
    # Reading it back must return exactly the same dict.
    assert cache.get(key) == {"text": "hello é", "stop_reason": "end_turn"}


def test_cache_key_provider_and_extra_isolation():
    # Base key with no provider/extra
    key_base = cache_key(**ARGS, run=1)
    # Key with provider
    key_qwen = cache_key(**ARGS, run=1, provider="qwen")
    # Key with quantization extra
    key_4bit = cache_key(**ARGS, run=1, provider="qwen", extra={"load_in_4bit": True})
    key_16bit = cache_key(**ARGS, run=1, provider="qwen", extra={"load_in_4bit": False})

    assert key_base != key_qwen
    assert key_qwen != key_4bit
    assert key_4bit != key_16bit
