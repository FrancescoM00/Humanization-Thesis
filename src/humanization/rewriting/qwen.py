"""Qwen instruction formatting and token decoding; generation is greedy."""
def format_input(tokenizer, system, text):
    return tokenizer.apply_chat_template([{"role":"system","content":system["prompt"]},{"role":"user","content":text}], tokenize=False, add_generation_prompt=True)

def generated_tokens(sequence, input_tokens):
    return sequence[input_tokens:]
