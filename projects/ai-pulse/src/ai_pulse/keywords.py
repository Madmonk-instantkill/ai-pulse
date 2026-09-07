"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

POSITIVE_KEYWORD_CLUSTERS: dict[str, list[str]] = {
    # 1. Large Language Models and Foundation Models
    "llm_foundation_models": [
        "large language model",
        "foundation model",
    ],
    # 2. Fine-tuning, LoRA, QLoRA, and post-training
    "fine_tuning_post_training": [
        "fine-tuning",
        "LoRA",
        "QLoRA",
        "post-training",
    ],
    # 3. RLHF, DPO, RLVR, and reinforcement learning for language models
    "rl_for_language_models": [
        "RLHF",
        "DPO",
        "RLVR",
        "reinforcement learning for language models",
    ],
    # 4. LLM agents, reasoning, tool use, and planning
    "agents_reasoning_planning": [
        "LLM agent",
        "reasoning",
        "tool use",
        "planning",
    ],
    # 5. RAG, context compression, and long-context models
    "rag_long_context": [
        "retrieval-augmented generation",
        "context compression",
        "long-context model",
    ],
    # 6. Multimodal LLMs, including vision-language and audio-language models
    "multimodal": [
        "multimodal large language model",
        "vision-language model",
        "audio-language model",
    ],
    # 7. Efficient training, inference optimization, and model compression
    "efficiency": [
        "efficient training",
        "inference optimization",
        "model compression",
    ],
}


def flatten_keyword_terms() -> list[str]:
    """Flatten POSITIVE_KEYWORD_CLUSTERS into one plain list of terms,
    regardless of which cluster they came from. Both the arXiv and
    OpenAlex tools build their own query syntax on top of this same list,
    so the two sources always search for the same topics."""
    return [term for cluster in POSITIVE_KEYWORD_CLUSTERS.values() for term in cluster]
