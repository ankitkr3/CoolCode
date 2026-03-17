"""Cool Code CLI — the smarter coding assistant.

Usage:
    coolcode                  # Interactive mode
    coolcode "fix the bug"    # One-shot task
    coolcode --provider minimax "write tests"  # Use specific provider
    coolcode --strategy quality "refactor auth"  # Use quality-first routing
    coolcode --workers 5 "complex feature"  # More parallel workers
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

import click
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from coolcode.agent.swarm import Swarm, SwarmResult
from coolcode.config import CoolCodeConfig
from coolcode.llm.provider import LLMProvider
from coolcode.memory.collective import CollectiveMemory
from coolcode.memory.scoped import ScopedMemory
from coolcode.prompts.system import build_system_prompt
from coolcode.tools import ALL_TOOLS

console = Console()
logger = logging.getLogger("coolcode")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_banner() -> None:
    banner = Text()
    banner.append("  ____            _    ____          _      \n", style="bold cyan")
    banner.append(" / ___|___   ___ | |  / ___|___   __| | ___ \n", style="bold cyan")
    banner.append("| |   / _ \\ / _ \\| | | |   / _ \\ / _` |/ _ \\\n", style="bold cyan")
    banner.append("| |__| (_) | (_) | | | |__| (_) | (_| |  __/\n", style="bold cyan")
    banner.append(" \\____\\___/ \\___/|_|  \\____\\___/ \\__,_|\\___|\n", style="bold cyan")
    banner.append("\n  Swarm-powered coding agent | v0.1.0\n", style="dim")
    console.print(banner)


def _print_stats(result: SwarmResult) -> None:
    stats = result.stats
    table = Table(title="Swarm Stats", show_header=False, border_style="dim")
    table.add_column("Key", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Workers spawned", str(stats["total_workers"]))
    table.add_row("Successful", str(stats["successful"]))
    table.add_row("Failed", str(stats["failed"]))
    table.add_row("Types used", ", ".join(stats["worker_types"]))
    table.add_row("Best worker", stats["best_worker"] or "N/A")
    table.add_row("Winning provider", stats.get("winning_provider", "N/A"))
    table.add_row("Confidence", f"{stats['best_confidence']:.0%}")
    table.add_row("Consensus", stats["consensus"])
    table.add_row("Avg latency", f"{stats['avg_latency_ms']:.0f}ms")
    # Show per-provider breakdown
    for provider, counts in stats.get("provider_breakdown", {}).items():
        table.add_row(f"  {provider}", f"{counts['success']}/{counts['total']} succeeded")
    console.print(table)


async def _run_task(task: str, config: CoolCodeConfig, strategy: str) -> None:
    """Execute a single task through the swarm."""
    llm_provider = LLMProvider(config, strategy=strategy)

    # Show available providers
    providers = llm_provider.available_providers
    console.print(f"[dim]Providers: {', '.join(providers)}[/dim]")
    console.print(f"[dim]Strategy: {strategy} | Workers: {config.swarm.num_workers} | "
                  f"Consensus: {config.swarm.consensus_algorithm}[/dim]")
    console.print()

    collective_memory = CollectiveMemory(config.memory.sqlite_path)

    swarm = Swarm(
        config=config,
        llm_provider=llm_provider,
        collective_memory=collective_memory,
    )

    start = time.monotonic()
    with console.status("[bold cyan]Swarm is working...[/bold cyan]", spinner="dots"):
        result = await swarm.execute(task, tools=ALL_TOOLS)
    elapsed = time.monotonic() - start

    # Display result
    console.print()
    console.print(Panel(Markdown(result.output), title="Result", border_style="green"))
    console.print()
    console.print(f"[dim]Completed in {elapsed:.1f}s[/dim]")
    _print_stats(result)

    # Save routing data
    swarm.task_router.save()


def _show_model_status(config: CoolCodeConfig) -> None:
    """Display which providers/models are currently active."""
    table = Table(title="Active Models", show_header=True, border_style="cyan")
    table.add_column("Provider", style="bold")
    table.add_column("Model")
    table.add_column("Status")
    for p in config.providers:
        status = "[green]active[/green]" if p.api_key else "[red]no API key[/red]"
        table.add_row(p.provider, p.model, status)
    if not config.providers:
        table.add_row("—", "—", "[red]No providers configured[/red]")
    console.print(table)


def _handle_model_command(args: str, config: CoolCodeConfig) -> None:
    """/model command — switch models on the fly.

    Usage:
        /model                      — show active models
        /model claude               — use only Claude
        /model minimax              — use only MiniMax
        /model both                 — use both (parallel racing)
        /model claude:claude-opus-4-6  — switch Claude to Opus
        /model minimax:MiniMax-M2.5-Lightning  — switch MiniMax to Lightning
    """
    args = args.strip()

    if not args:
        _show_model_status(config)
        console.print()
        console.print("[dim]Usage: /model <claude|minimax|both> or /model <provider>:<model-name>[/dim]")
        return

    if args == "both":
        # Re-enable both providers from env
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        minimax_key = os.getenv("MINIMAX_API_KEY", "")
        from coolcode.config import LLMConfig
        config.providers = []
        if anthropic_key:
            config.providers.append(LLMConfig(
                provider="claude",
                model=os.getenv("COOLCODE_CLAUDE_MODEL", "claude-sonnet-4-6"),
                api_key=anthropic_key,
            ))
        if minimax_key:
            config.providers.append(LLMConfig(
                provider="minimax",
                model=os.getenv("COOLCODE_MINIMAX_MODEL", "MiniMax-M2.5"),
                api_key=minimax_key,
                group_id=os.getenv("MINIMAX_GROUP_ID", ""),
            ))
        console.print("[green]Switched to both providers (parallel racing)[/green]")
        _show_model_status(config)
        return

    if ":" in args:
        # Switch a specific provider's model: e.g., "claude:claude-opus-4-6"
        provider_name, model_name = args.split(":", 1)
        for p in config.providers:
            if p.provider == provider_name:
                old_model = p.model
                p.model = model_name
                console.print(f"[green]{provider_name}: {old_model} → {model_name}[/green]")
                return
        console.print(f"[red]Provider '{provider_name}' not found. Add its API key first.[/red]")
        return

    # Single provider name: claude or minimax — keep only that one
    target = args.lower()
    matching = [p for p in config.providers if p.provider == target]
    if not matching:
        console.print(f"[red]Provider '{target}' not configured. Set its API key.[/red]")
        return
    config.providers = matching
    console.print(f"[green]Switched to {target} only[/green]")
    _show_model_status(config)


def _interactive_loop(config: CoolCodeConfig, strategy: str) -> None:
    """Run the interactive REPL."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    history_path = os.path.expanduser("~/.coolcode/history")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    session = PromptSession(history=FileHistory(history_path))

    _print_banner()

    # Auto-detect: use whatever providers have API keys
    active = [p.provider for p in config.providers if p.api_key]
    mode = "parallel racing" if len(active) > 1 else active[0] if active else "none"
    console.print(f"[dim]Active providers: {', '.join(active)} ({mode}) | Strategy: {strategy}[/dim]")
    console.print("[dim]Commands: /model, /stats, /help, quit[/dim]")
    console.print()

    while True:
        try:
            user_input = session.prompt("Cool Code > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        # /model command
        if user_input.startswith("/model"):
            _handle_model_command(user_input[6:], config)
            continue

        if user_input == "/stats":
            provider = LLMProvider(config, strategy=strategy)
            stats = provider.get_stats()
            for key, s in stats.items():
                console.print(f"  {key}: {s}")
            continue
        if user_input == "/help":
            console.print("[bold]Commands:[/bold]")
            console.print("  /model              — Show/switch active models")
            console.print("  /model claude       — Use only Claude")
            console.print("  /model minimax      — Use only MiniMax")
            console.print("  /model both         — Use both (parallel racing)")
            console.print("  /model claude:claude-opus-4-6  — Switch Claude model")
            console.print("  /stats              — Show provider performance stats")
            console.print("  /help               — Show this help")
            console.print("  quit                — Exit Cool Code")
            continue

        try:
            asyncio.run(_run_task(user_input, config, strategy))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.exception("Task execution failed")

        console.print()


@click.command()
@click.argument("task", required=False)
@click.option("--provider", "-p", type=click.Choice(["claude", "minimax"]), help="Preferred LLM provider")
@click.option("--strategy", "-s", default="cost", type=click.Choice(["cost", "quality", "fast"]), help="Routing strategy")
@click.option("--workers", "-w", default=3, type=int, help="Number of parallel workers")
@click.option("--consensus", "-c", default="weighted", type=click.Choice(["majority", "weighted", "raft", "byzantine", "gossip"]), help="Consensus algorithm")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(
    task: str | None,
    provider: str | None,
    strategy: str,
    workers: int,
    consensus: str,
    verbose: bool,
) -> None:
    """Cool Code — the smarter CLI coding agent with swarm intelligence."""
    _setup_logging(verbose)

    config = CoolCodeConfig.from_env(project_dir=os.getcwd())
    config.swarm.num_workers = workers
    config.swarm.consensus_algorithm = consensus
    config.verbose = verbose

    if provider:
        os.environ["COOLCODE_DEFAULT_PROVIDER"] = provider

    if task:
        # One-shot mode
        try:
            asyncio.run(_run_task(task, config, strategy))
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
    else:
        # Interactive mode
        _interactive_loop(config, strategy)


if __name__ == "__main__":
    main()
