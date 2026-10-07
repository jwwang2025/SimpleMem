"""
SimpleMem — Efficient Lifelong Memory for LLM Agents (Text Memory Core)

Usage:
    from simplemem import SimpleMem

    mem = SimpleMem()
    mem.add_dialogue("Alice", "Let's meet at 2pm", "2025-11-15T14:30:00")
    mem.finalize()
    answer = mem.ask("When will they meet?")
"""

from simplemem.text.system import SimpleMemSystem as SimpleMem, create_system as create
from simplemem.config import Config, load_config


def list_modes():
    """Return available memory modes."""
    return {"text": "Single-modal text memory with semantic lossless compression"}


__version__ = "0.3.0"
__all__ = ["SimpleMem", "create", "list_modes", "Config", "load_config"]
