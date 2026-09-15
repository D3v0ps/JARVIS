"""Tool layer for JARVIS.

The package is deliberately import-light: it contains no side effects beyond the
registry itself, so ``import jarvis.tools`` is safe on any platform.

Layout
------
``base``        dataclasses shared by every tool (``ToolResult``, ``ToolSpec``, ...).
``registry``    the ``@tool`` decorator, the global ``REGISTRY`` and ``load_all()``.
``safety``      command blocklist, tier escalation and spoken confirm/cancel parsing.
``dispatcher``  runtime execution of a single tool call.
``*_tools``     the actual tools; they are discovered and imported by ``load_all()``.

Concrete tool modules import Windows-only libraries lazily, inside the function
that needs them, so this package stays importable on Linux CI.
"""
