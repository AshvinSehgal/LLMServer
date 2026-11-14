"""Natural chat prompts for speculative-decoding benchmarks.

The throughput benchmark's filler text repeats itself, which a small draft model predicts almost perfectly, so it would
overstate speculative decoding. These are varied, ordinary requests (explanation, code, math, writing, extraction).
"""
QUESTIONS = [
    "Explain in a few sentences how a refrigerator works.",
    "Write a short Python function that checks whether a number is prime, with a docstring.",
    "What are three differences between TCP and UDP? Answer as a short list.",
    "A train leaves at 3:15 pm and travels 210 km at 70 km/h. When does it arrive? Show the steps.",
    "Write a four-line poem about autumn rain.",
    "Summarize the plot of Romeo and Juliet in a short paragraph.",
    "Translate into French: 'The library closes at eight, so please return your books before then.'",
    "Give me a simple recipe for pancakes, with ingredients and steps.",
]


def chat_token_ids(tokenizer, question):
    """Prompt token ids with the model's chat template (thinking off for models that have a switch)."""
    messages = [{"role": "user", "content": question}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"]
