from __future__ import annotations

import asyncio
import logging
from typing import List

from .node_manager import NodeManager
from .network_node import NetworkMessage

try:
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.console import Console
    RICH_AVAILABLE = True
    console = Console()
except Exception:  # pragma: no cover - rich is optional
    RICH_AVAILABLE = False
    console = None

logger = logging.getLogger(__name__)


class InteractiveShell:
    """Interactive command shell for the node."""

    def __init__(self, manager: NodeManager) -> None:
        self.manager = manager
        self.commands = {
            "help": self.cmd_help,
            "peers": self.cmd_peers,
            "dns": self.cmd_dns,
            "send": self.cmd_send,
            "stats": self.cmd_stats,
            "loglevel": self.cmd_loglevel,
            "quit": self.cmd_quit,
        }

    async def run(self) -> None:
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]Enhanced CSP Node Interactive Shell[/bold cyan]")
            console.print("Type 'help' for available commands\n")
        else:
            print("\nEnhanced CSP Node Interactive Shell")
            print("Type 'help' for available commands\n")

        while not self.manager.shutdown_event.is_set():
            try:
                if RICH_AVAILABLE:
                    command = await asyncio.get_event_loop().run_in_executor(None, Prompt.ask, "[bold]csp>[/bold]")
                else:
                    command = await asyncio.get_event_loop().run_in_executor(None, input, "csp>: ")

                if not command:
                    continue

                parts = command.strip().split()
                if not parts:
                    continue

                cmd = parts[0].lower()
                args = parts[1:]

                if cmd in self.commands:
                    await self.commands[cmd](args)
                else:
                    print(f"Unknown command: {cmd}")
            except (EOFError, KeyboardInterrupt):
                print("\nUse 'quit' to exit")
            except Exception as exc:  # pragma: no cover - best effort
                print(f"Error: {exc}")

    async def cmd_help(self, args: List[str]) -> None:  # noqa: ARG002
        help_text = """\
Available commands:
  help              - Show this help message
  peers             - List connected peers
  dns <name>        - Resolve .web4ai domain
  send <peer> <msg> - Send message to peer
  stats             - Show node statistics
  loglevel <level>  - Set logging level
  quit              - Exit the shell
"""
        print(help_text)

    async def cmd_peers(self, args: List[str]) -> None:  # noqa: ARG002
        try:
            peers = self.manager.network.get_peers() if hasattr(self.manager.network, "get_peers") else []
        except Exception:
            peers = []

        if not peers:
            print("No connected peers")
            return

        if RICH_AVAILABLE:
            table = Table(title="Connected Peers")
            table.add_column("Peer ID", style="cyan")
            table.add_column("Address", style="green")
            table.add_column("Latency", style="yellow")
            table.add_column("Reputation", style="blue")
            for peer in peers:
                table.add_row(
                    str(getattr(peer, "id", "unknown"))[:16] + "...",
                    f"{getattr(peer, 'address', 'unknown')}:{getattr(peer, 'port', 0)}",
                    f"{getattr(peer, 'latency', 0):.2f}ms" if getattr(peer, 'latency', None) else "N/A",
                    f"{getattr(peer, 'reputation', 0):.2f}",
                )
            console.print(table)
        else:
            print(f"\nConnected peers ({len(peers)}):")
            for peer in peers:
                print(f"  {getattr(peer, 'id', 'unknown')}: {getattr(peer, 'address', 'unknown')}:{getattr(peer, 'port', 0)}")

    async def cmd_dns(self, args: List[str]) -> None:
        if not args:
            print("Usage: dns <name>")
            print("       dns list              - List all DNS records (genesis only)")
            print("       dns register <name> <addr> - Register DNS name (genesis only)")
            return

        if not hasattr(self.manager.network, "dns_overlay") or not self.manager.network.dns_overlay:
            print("DNS overlay not available")
            return

        if args[0] == "list" and self.manager.is_genesis:
            try:
                if hasattr(self.manager.network.dns_overlay, "list_records"):
                    records = await self.manager.network.dns_overlay.list_records()
                else:
                    records = getattr(self.manager.network.dns_overlay, "records", {})
                if RICH_AVAILABLE:
                    table = Table(title="DNS Records")
                    table.add_column("Domain", style="cyan")
                    table.add_column("Address", style="green")
                    for domain, addr in records.items():
                        table.add_row(domain, addr)
                    console.print(table)
                else:
                    print("\nDNS Records:")
                    for domain, addr in records.items():
                        print(f"  {domain} -> {addr}")
            except Exception as exc:
                print(f"Failed to list DNS records: {exc}")
            return

        if args[0] == "register" and len(args) >= 3 and self.manager.is_genesis:
            domain = args[1]
            addr = " ".join(args[2:])
            try:
                if hasattr(self.manager.network.dns_overlay, "register"):
                    await self.manager.network.dns_overlay.register(domain, addr)
                    print(f"Registered: {domain} -> {addr}")
                else:
                    print("DNS registration not available")
            except Exception as exc:
                print(f"Failed to register: {exc}")
            return

        name = args[0]
        try:
            if hasattr(self.manager.network.dns_overlay, "resolve"):
                result = await self.manager.network.dns_overlay.resolve(name)
                print(f"{name} -> {result}")
            else:
                print("DNS resolution not available")
        except Exception as exc:
            print(f"Failed to resolve {name}: {exc}")

    async def cmd_send(self, args: List[str]) -> None:
        if len(args) < 2:
            print("Usage: send <peer_id> <message>")
            return

        peer_id = args[0]
        message = " ".join(args[1:])

        try:
            if hasattr(self.manager.network, "send_message"):
                success = await self.manager.network.send_message(peer_id, {"content": message, "type": "chat"})
                if success:
                    print(f"Message sent to {peer_id}")
                else:
                    print(f"Failed to send message to {peer_id}")
            else:
                print("Message sending not implemented")
        except Exception as exc:
            print(f"Failed to send message: {exc}")

    async def cmd_stats(self, args: List[str]) -> None:  # noqa: ARG002
        metrics = await self.manager.collect_metrics()
        if RICH_AVAILABLE:
            table = Table(title="Node Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            for key, value in metrics.items():
                if key == "uptime":
                    value = f"{value:.0f}s"
                elif key in ["bandwidth_in", "bandwidth_out"]:
                    value = f"{value / 1024 / 1024:.2f} MB"
                table.add_row(key.replace("_", " ").title(), str(value))
            console.print(table)
        else:
            print("\nNode Statistics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")

    async def cmd_loglevel(self, args: List[str]) -> None:
        if not args:
            print("Usage: loglevel <debug|info|warning|error>")
            return
        level = args[0].upper()
        try:
            logging.getLogger().setLevel(getattr(logging, level))
            print(f"Log level set to {level}")
        except AttributeError:
            print(f"Invalid log level: {level}")

    async def cmd_quit(self, args: List[str]) -> None:  # noqa: ARG002
        print("Exiting shell...")
        self.manager.shutdown_event.set()
