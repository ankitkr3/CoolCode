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
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from coolcode.agent.swarm import Swarm, SwarmResult
from coolcode.config import AVAILABLE_MODELS, CoolCodeConfig, LLMConfig
from coolcode.llm.provider import LLMProvider
from coolcode.memory.collective import CollectiveMemory
from coolcode.prompts.system import build_system_prompt
from coolcode.status import StatusTracker
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


# ---------------------------------------------------------------------------
# Onboarding — first-time setup
# ---------------------------------------------------------------------------

def _onboarding(config: CoolCodeConfig) -> CoolCodeConfig:
    """Interactive setup when no providers are configured.

    Asks user to:
    1. Choose provider(s)
    2. Select model(s)
    3. Enter API key(s)

    Persists to ~/.coolcode/config.json so it never asks again.
    """
    console.print()
    console.print(Panel(
        "[bold]Welcome to Cool Code![/bold]\n\n"
        "Let's set up your LLM provider(s). This only happens once —\n"
        "your config is saved to [cyan]~/.coolcode/config.json[/cyan]\n"
        "and persists across sessions. Use [cyan]/model[/cyan] anytime to change.",
        border_style="cyan",
    ))
    console.print()

    # Step 1: Choose provider(s)
    console.print("[bold]Step 1:[/bold] Which provider(s) do you want to use?\n")
    console.print("  [cyan]1[/cyan]  Claude (Anthropic)        — Best code quality")
    console.print("  [cyan]2[/cyan]  MiniMax M2.5              — Cheapest, very fast")
    console.print("  [cyan]3[/cyan]  Both (parallel racing)    — Best of both worlds")
    console.print()

    choice = ""
    while choice not in ("1", "2", "3"):
        try:
            choice = console.input("[bold]Choose (1/2/3): [/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Setup cancelled.[/dim]")
            sys.exit(0)

    selected_providers: list[str] = []
    if choice == "1":
        selected_providers = ["claude"]
    elif choice == "2":
        selected_providers = ["minimax"]
    elif choice == "3":
        selected_providers = ["claude", "minimax"]

    providers: list[LLMConfig] = []

    for provider_name in selected_providers:
        console.print()
        models = AVAILABLE_MODELS[provider_name]

        # Step 2: Choose model
        console.print(f"[bold]Step 2:[/bold] Select a [cyan]{provider_name}[/cyan] model:\n")
        for i, m in enumerate(models, 1):
            console.print(f"  [cyan]{i}[/cyan]  {m['name']:30s} ({m['tier']})")
        console.print()

        model_choice = ""
        valid = [str(i) for i in range(1, len(models) + 1)]
        while model_choice not in valid:
            try:
                model_choice = console.input(f"[bold]Choose (1-{len(models)}) [default: 1]: [/bold]").strip()
                if not model_choice:
                    model_choice = "1"
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Setup cancelled.[/dim]")
                sys.exit(0)

        selected_model = models[int(model_choice) - 1]

        # Step 3: API key
        console.print()
        if provider_name == "claude":
            console.print("[dim]Get your key at: https://console.anthropic.com/settings/keys[/dim]")
        else:
            console.print("[dim]Get your key at: https://api.minimax.chat[/dim]")

        api_key = ""
        while not api_key:
            try:
                api_key = console.input(f"[bold]Enter {provider_name} API key: [/bold]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Setup cancelled.[/dim]")
                sys.exit(0)

        group_id = ""
        if provider_name == "minimax":
            try:
                group_id = console.input("[bold]Enter MiniMax Group ID: [/bold]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Setup cancelled.[/dim]")
                sys.exit(0)

        providers.append(LLMConfig(
            provider=provider_name,
            model=selected_model["id"],
            api_key=api_key,
            group_id=group_id,
        ))

    config.providers = providers
    config.save()

    console.print()
    console.print("[green]Setup complete! Config saved to ~/.coolcode/config.json[/green]")
    _show_model_status(config)
    console.print()
    return config


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

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
    """/model command — switch models on the fly. Changes are persisted.

    Usage:
        /model                      — show active models
        /model setup                — re-run onboarding
        /model claude               — use only Claude
        /model minimax              — use only MiniMax
        /model both                 — use both (parallel racing)
        /model claude:claude-opus-4-6  — switch Claude to Opus
    """
    args = args.strip()

    if not args:
        _show_model_status(config)
        console.print()
        console.print("[dim]Usage: /model <claude|minimax|both|setup> or /model <provider>:<model-name>[/dim]")
        return

    if args == "setup":
        _onboarding(config)
        return

    if args == "both":
        # Ensure both providers exist in config
        has_claude = any(p.provider == "claude" for p in config.providers)
        has_minimax = any(p.provider == "minimax" for p in config.providers)

        if not has_claude or not has_minimax:
            console.print("[yellow]Missing a provider. Let's set up the missing one.[/yellow]")
            _onboarding(config)
            return

        console.print("[green]Switched to both providers (parallel racing)[/green]")
        _show_model_status(config)
        config.save()
        return

    if ":" in args:
        provider_name, model_name = args.split(":", 1)
        for p in config.providers:
            if p.provider == provider_name:
                old_model = p.model
                p.model = model_name
                config.save()
                console.print(f"[green]{provider_name}: {old_model} -> {model_name} (saved)[/green]")
                return
        console.print(f"[red]Provider '{provider_name}' not configured. Run /model setup[/red]")
        return

    # Single provider: keep only that one active but don't delete the other from saved config
    target = args.lower()
    matching = [p for p in config.providers if p.provider == target]
    if not matching:
        console.print(f"[red]Provider '{target}' not configured. Run /model setup[/red]")
        return
    # Store the full list for /model both later, but use only the selected one
    config.providers = matching
    config.save()
    console.print(f"[green]Switched to {target} only (saved)[/green]")
    _show_model_status(config)


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------

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
    for provider, counts in stats.get("provider_breakdown", {}).items():
        table.add_row(f"  {provider}", f"{counts['success']}/{counts['total']} succeeded")
    console.print(table)


async def _run_task(task: str, config: CoolCodeConfig, strategy: str) -> None:
    """Execute a single task through the swarm with live progress display."""
    if not config.providers:
        console.print("[red]No providers configured. Run /model setup[/red]")
        return

    llm_provider = LLMProvider(config, strategy=strategy)
    status_tracker = StatusTracker()

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
        status_tracker=status_tracker,
    )

    start = time.monotonic()

    # Run swarm in background, display live progress in foreground
    swarm_task = asyncio.create_task(swarm.execute(task, tools=ALL_TOOLS))

    from rich.live import Live
    from rich.text import Text as RichText

    log_lines: list[str] = []

    with Live(console=console, refresh_per_second=8) as live:
        hindi_timer = time.monotonic()
        hindi_msg = status_tracker.next_hindi()

        while not swarm_task.done():
            update = await status_tracker.get(timeout=0.3)

            if update:
                # Format: source icon + action + detail
                icons = {
                    "swarm": "[bold yellow]>[/bold yellow]",
                    "queen": "[bold magenta]Q[/bold magenta]",
                    "router": "[bold blue]R[/bold blue]",
                }
                icon = icons.get(update.source, "[bold cyan]W[/bold cyan]")
                elapsed_so_far = time.monotonic() - start
                line = f"  {icon} [{elapsed_so_far:5.1f}s] [bold]{update.source}[/bold] {update.action}: [dim]{update.detail}[/dim]"
                log_lines.append(line)

            # Rotate Hindi message every 4 seconds
            if time.monotonic() - hindi_timer > 4.0:
                hindi_msg = status_tracker.next_hindi()
                hindi_timer = time.monotonic()

            # Build display
            display = RichText()
            display.append(f"  {hindi_msg}\n\n", style="bold yellow")

            # Show last 12 log lines
            visible = log_lines[-12:]
            display_text = f"  {hindi_msg}\n\n" + "\n".join(visible)
            live.update(
                Panel(
                    display_text,
                    title="[bold cyan]Cool Code working...[/bold cyan]",
                    border_style="cyan",
                    subtitle=f"[dim]{time.monotonic() - start:.1f}s elapsed[/dim]",
                )
            )

    result = swarm_task.result()
    elapsed = time.monotonic() - start

    # Show final log
    console.print()
    if log_lines:
        console.print(Panel(
            "\n".join(log_lines),
            title="[dim]Activity Log[/dim]",
            border_style="dim",
        ))

    console.print()
    console.print(Panel(Markdown(result.output), title="Result", border_style="green"))
    console.print()
    console.print(f"[dim]Completed in {elapsed:.1f}s[/dim]")
    _print_stats(result)

    swarm.task_router.save()


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def _interactive_loop(config: CoolCodeConfig, strategy: str) -> None:
    """Run the interactive REPL."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    history_path = os.path.expanduser("~/.coolcode/history")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    session = PromptSession(history=FileHistory(history_path))

    _print_banner()

    # If no providers configured, run onboarding
    if not config.providers:
        config = _onboarding(config)

    # Show active state
    active = [p.provider for p in config.providers if p.api_key]
    if len(active) > 1:
        mode = "parallel racing"
    elif active:
        mode = active[0]
    else:
        mode = "none"
    console.print(f"[dim]Active: {', '.join(active)} ({mode}) | Strategy: {strategy}[/dim]")
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

        if user_input.startswith("/model"):
            _handle_model_command(user_input[6:], config)
            continue

        if user_input == "/stats":
            if not config.providers:
                console.print("[yellow]No providers configured. Run /model setup[/yellow]")
                continue
            provider = LLMProvider(config, strategy=strategy)
            stats = provider.get_stats()
            console.print("[bold]Provider Stats:[/bold]")
            for key, s in stats.items():
                console.print(f"  {key}: {s}")

            from coolcode.learner import WorkflowLearner

            learner = WorkflowLearner(
                persist_path=str(Path.home() / ".coolcode" / "learnings.json")
            )
            lstats = learner.stats
            console.print(f"\n[bold]Learnings:[/bold]")
            console.print(f"  Patterns learned: {lstats['patterns_learned']}")
            console.print(f"  Total executions: {lstats['total_executions']}")
            if lstats.get('avg_success_rate'):
                console.print(f"  Avg success rate: {lstats['avg_success_rate']}")
            continue

        if user_input == "/help":
            console.print("[bold]Commands:[/bold]")
            console.print("  /model              — Show active models")
            console.print("  /model setup        — Re-run provider setup")
            console.print("  /model claude       — Switch to Claude only")
            console.print("  /model minimax      — Switch to MiniMax only")
            console.print("  /model both         — Both providers (parallel racing)")
            console.print("  /model claude:claude-opus-4-6  — Switch specific model")
            console.print("  /stats              — Provider stats, cache, and learnings")
            console.print("  /help               — Show this help")
            console.print("  quit                — Exit Cool Code")
            continue

        try:
            asyncio.run(_run_task(user_input, config, strategy))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.exception("Task execution failed")

        console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
        # One-shot mode — must have providers
        if not config.providers:
            console.print("[red]No providers configured. Run `coolcode` first to set up.[/red]")
            sys.exit(1)
        try:
            asyncio.run(_run_task(task, config, strategy))
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
    else:
        # Interactive mode — onboarding runs if needed
        _interactive_loop(config, strategy)


if __name__ == "__main__":
    main()
