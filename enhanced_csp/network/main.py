#!/usr/bin/env python3
"""Enhanced CSP Network entry script."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from .cli import parse_args
from .node_manager import NodeManager
from .shell import InteractiveShell
from .status_server import StatusServer

try:
    from rich.console import Console
    RICH_AVAILABLE = True
    console = Console()
except Exception:  # pragma: no cover - optional dependency
    RICH_AVAILABLE = False
    console = None

try:
    from aiohttp import web  # noqa:F401
    AIOHTTP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


async def run_main(args: argparse.Namespace) -> None:
    manager = NodeManager(args)
    shell = InteractiveShell(manager) if not args.no_shell else None
    status_server = None

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(handle_signal(s, manager)))

    try:
        await manager.initialize()

        if not args.no_status and AIOHTTP_AVAILABLE:
            server = StatusServer(manager, args.status_port)
            status_server = await server.start()
        elif not AIOHTTP_AVAILABLE and not args.no_status:
            manager.logger.warning("aiohttp not available - status server disabled")

        if shell and not args.no_shell:
            await shell.run()
        else:
            await manager.shutdown_event.wait()
    except Exception as exc:
        manager.logger.error(f"Fatal error: {exc}", exc_info=True)
        raise
    finally:
        if status_server:
            await status_server.cleanup()
        await manager.shutdown()


def handle_signal(sig: signal.Signals, manager: NodeManager) -> None:
    manager.logger.info(f"Received signal {sig.value}")
    manager.shutdown_event.set()


async def main() -> None:
    args = parse_args()

    if RICH_AVAILABLE and not args.no_shell:
        console.print("[bold cyan]Enhanced CSP Network Node[/bold cyan]")
        console.print("[dim]Version 1.0.0[/dim]")
        if args.genesis:
            console.print("[bold yellow]🌟 GENESIS NODE - First in the network[/bold yellow]")
        console.print()

    await run_main(args)


if __name__ == "__main__":
    asyncio.run(main())
