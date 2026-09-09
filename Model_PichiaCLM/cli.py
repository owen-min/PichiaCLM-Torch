"""Backward-compatible CLI module.

Prefer: python -m Model_PichiaCLM.interfaces.cli
"""

from .interfaces.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
