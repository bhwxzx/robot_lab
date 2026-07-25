#!/usr/bin/env python3
"""Backward-compatible entry point for the profile-driven training log parser."""

from summarize_training_log import main, parse_log

__all__ = ["main", "parse_log"]


if __name__ == "__main__":
    raise SystemExit(main())
