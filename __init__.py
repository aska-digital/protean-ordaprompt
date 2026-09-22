"""protean-ordaprompt — Hermes plugin wrapper.

The agent-half payload lives in the bundled ``ordaprompt_router`` package and its
CLI (``python3 -m ordaprompt_router.cli``). This wrapper declares no tools, no
hooks, no middleware and no settings: it exists so the plugin has a loadable
Python surface and so the router stays importable from an enabled install
without any path probing.

Import discipline: standard library only; the router package itself is imported
lazily so this module stays importable on a bare interpreter.
"""

from __future__ import annotations

__version__ = "1.1.0"

PLUGIN_NAME = "protean-ordaprompt"


def register(ctx) -> None:
    """Satisfy the directory-plugin load contract (plugins_loader.py register probe).

    protean-ordaprompt declares no tools, no hooks, no middleware and no settings:
    its entire agent-half surface is the bundled ``ordaprompt_router`` package and
    the explicit CLI (``python3 -m ordaprompt_router.cli``). Nothing subscribes to
    the session loop, so there is nothing for register() to add — a no-op register()
    is the correct and complete registration for this package.
    """
    return None
