"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

import tiktoken

from ai_pulse.schemas import DocumentText

# cl100k_base is OpenAI's tokenizer, not Gemini's -- there is no exact
# Gemini tokenizer available this way. It is still the standard,
# widely-used approximation for "roughly how many tokens will this cost",
# close enough for chunk-sizing decisions even though it won't be a
# perfectly exact count for the model we actually call.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count how many tokens a piece of text costs, using cl100k_base as
    a close, standard approximation."""
    return len(_ENCODING.encode(text))


def annotate_token_counts(document: DocumentText) -> DocumentText:
    """Return a copy of this document with every page's token_count
    filled in. Plain Python, no LLM involved -- counting tokens is a
    mechanical lookup, not a judgment call."""
    annotated_pages = [
        page.model_copy(update={"token_count": count_tokens(page.text)})
        for page in document.pages
    ]
    return document.model_copy(update={"pages": annotated_pages})
