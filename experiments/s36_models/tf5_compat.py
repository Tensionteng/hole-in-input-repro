"""Shims that let the trust_remote_code time-series LLMs (TimeMoE, Sundial) run under
transformers 5.x. Their vendored generation mixins were written against 4.3x and call three
cache/generation helpers that have since been removed or changed signature. Each shim
restores the 4.x behaviour exactly; none of them touches the forward pass, so the forecasts
are the checkpoints' own.
"""
from transformers.cache_utils import DynamicCache
from transformers.generation.utils import GenerationMixin

if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda s: s.get_seq_length())
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda s: None
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda s, new_len, layer_idx=0: s.get_seq_length(layer_idx)
if not hasattr(GenerationMixin, "_extract_past_from_model_output"):
    # 4.3x returned the cache itself; 5.x returns (name, cache) and the vendored mixins
    # assign the result straight into model_kwargs["past_key_values"].
    GenerationMixin._extract_past_from_model_output = (
        lambda self, outputs, standardize_cache_format=False:
        getattr(outputs, "past_key_values", None))
