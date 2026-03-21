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
import warnings
from pathlib import Path

# Suppress HuggingFace tokenizers fork warning (we use sentence-transformers for embeddings)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Suppress sklearn/numpy version mismatch warning from sentence-transformers
warnings.filterwarnings("ignore", message=".*A NumPy version.*", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from coolcode.agent.swarm import Swarm, SwarmResult
from coolcode.config import AVAILABLE_MODELS, CoolCodeConfig, LLMConfig
from coolcode.cost import CostTracker
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


def _get_version() -> str:
    """Read version from package metadata, falling back to pyproject.toml."""
    try:
        from importlib.metadata import version
        return version("coolcode")
    except Exception:
        pass
    # Fallback: read pyproject.toml directly
    try:
        toml_path = Path(__file__).parent.parent / "pyproject.toml"
        for line in toml_path.read_text().splitlines():
            if line.strip().startswith("version"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return "dev"


def _print_banner() -> None:
    ver = _get_version()
    banner = Text()
    banner.append("  ____            _    ____          _      \n", style="bold cyan")
    banner.append(" / ___|___   ___ | |  / ___|___   __| | ___ \n", style="bold cyan")
    banner.append("| |   / _ \\ / _ \\| | | |   / _ \\ / _` |/ _ \\\n", style="bold cyan")
    banner.append("| |__| (_) | (_) | | | |__| (_) | (_| |  __/\n", style="bold cyan")
    banner.append(" \\____\\___/ \\___/|_|  \\____\\___/ \\__,_|\\___|\n", style="bold cyan")
    banner.append(f"\n  Swarm-powered coding agent | v{ver}\n", style="dim")
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
    console.print("  [cyan]2[/cyan]  MiniMax (M2.5 / M2.7)    — Fast & affordable")
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
    # Cost info
    if stats.get("task_cost_usd", 0) > 0:
        table.add_row("Task cost", f"${stats['task_cost_usd']:.4f}")
    if stats.get("input_tokens", 0) > 0:
        table.add_row("Tokens", f"{stats['input_tokens']:,} in / {stats['output_tokens']:,} out")
    for provider, counts in stats.get("provider_breakdown", {}).items():
        table.add_row(f"  {provider}", f"{counts['success']}/{counts['total']} succeeded")
    console.print(table)


def _create_swarm(config: CoolCodeConfig, strategy: str, cost_tracker: CostTracker | None = None) -> Swarm:
    """Create a Swarm instance that persists across the session."""
    llm_provider = LLMProvider(config, strategy=strategy)
    collective_memory = CollectiveMemory(config.memory.sqlite_path)
    return Swarm(
        config=config,
        llm_provider=llm_provider,
        collective_memory=collective_memory,
        status_tracker=StatusTracker(),
        cost_tracker=cost_tracker or CostTracker(),
    )


def _flush_stdin() -> None:
    """Flush any pending bytes from stdin.

    Rich Live sends cursor position queries (CPR: \\x1b[6n]) to the terminal.
    The terminal responds with \\x1b[<row>;<col>R which can leak into
    prompt-toolkit's input buffer, appearing as garbage like '[63;1R'.
    This drains all pending bytes after Rich Live exits.
    """
    import select
    try:
        import termios
        # Only flush if stdin is a real terminal
        if not sys.stdin.isatty():
            return
        # Drain all pending bytes (non-blocking)
        while select.select([sys.stdin], [], [], 0.0)[0]:
            os.read(sys.stdin.fileno(), 4096)
    except (ImportError, OSError, ValueError):
        pass  # Not a real terminal or stdin closed


def _start_keyboard_thread(swarm: Swarm, log_lines: list[str], cancel_event: asyncio.Event, loop: asyncio.AbstractEventLoop):
    """Start a background thread for keyboard input (ESC and text injection).

    Runs in a separate thread so it doesn't interfere with Rich Live rendering.
    Uses termios cbreak mode only in the reader thread.
    """
    import threading

    input_buffer: list[str] = []

    def _reader():
        import select
        try:
            import termios
            import tty
        except ImportError:
            return  # Not a real terminal

        fd = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
        except termios.error:
            return

        try:
            tty.setcbreak(fd)
            while not cancel_event.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = os.read(fd, 1).decode("utf-8", errors="ignore")
                    if ch == '\x1b':  # ESC
                        cancel_event.set()
                        swarm.cancel()
                        log_lines.append(
                            "  [bold red]✕[/bold red] [bold]user[/bold] cancelled: ESC pressed — stopping gracefully..."
                        )
                    elif ch in ('\r', '\n'):
                        if input_buffer:
                            user_text = "".join(input_buffer).strip()
                            if user_text:
                                swarm.inject_context(user_text)
                                log_lines.append(
                                    f"  [bold yellow]+[/bold yellow] [bold]user[/bold] added: [dim]{user_text[:80]}[/dim]"
                                )
                            input_buffer.clear()
                    elif ch == '\x7f':  # Backspace
                        if input_buffer:
                            input_buffer.pop()
                    elif ch.isprintable():
                        input_buffer.append(ch)
        except Exception:
            pass
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return t


async def _run_task(task: str, swarm: Swarm) -> None:
    """Execute a single task through a persistent swarm with live progress display."""
    # Refresh the status tracker for this task (new queue, same swarm)
    swarm.status = StatusTracker()
    status_tracker = swarm.status

    providers = swarm.llm_provider.available_providers
    goal_text = f" | Goal: {swarm.goal}" if swarm.goal != "general" else ""
    console.print(f"[dim]Providers: {', '.join(providers)}[/dim]")
    console.print(f"[dim]Strategy: {swarm.llm_provider.strategy} | Workers: {swarm.config.swarm.num_workers} | "
                  f"Consensus: {swarm.config.swarm.consensus_algorithm}{goal_text}[/dim]")
    console.print(f"[dim]ESC = cancel  |  type + Enter = add context[/dim]")
    console.print()

    start = time.monotonic()

    # Run swarm in background, display live progress in foreground
    swarm_task = asyncio.create_task(swarm.execute(task, tools=ALL_TOOLS))

    from rich.live import Live

    log_lines: list[str] = []
    cancel_event = asyncio.Event()

    # Start keyboard listener in a background thread (avoids interfering with Rich Live)
    loop = asyncio.get_event_loop()
    kb_thread = _start_keyboard_thread(swarm, log_lines, cancel_event, loop)

    hindi_timer = time.monotonic()
    hindi_msg = status_tracker.next_hindi()

    # Show the Live panel immediately with a loading message (no delay)
    initial_display = f"  [bold yellow]{hindi_msg}[/bold yellow]\n\n  [dim]Starting...[/dim]"
    with Live(
        Panel(
            initial_display,
            title="[bold cyan]⚡ Cool Code[/bold cyan]",
            border_style="cyan",
            subtitle="[dim]0.0s | ESC=cancel | type+Enter=add info[/dim]",
            padding=(1, 2),
        ),
        console=console,
        refresh_per_second=4,
        transient=True,
    ) as live:
        while not swarm_task.done():
            update = await status_tracker.get(timeout=0.25)

            if update:
                icons = {
                    "swarm": "[bold yellow]⚡[/bold yellow]",
                    "queen": "[bold magenta]♛[/bold magenta]",
                    "router": "[bold blue]⇢[/bold blue]",
                    "goal": "[bold green]◎[/bold green]",
                    "user": "[bold yellow]✎[/bold yellow]",
                }
                icon = icons.get(update.source, "[bold cyan]●[/bold cyan]")
                elapsed_so_far = time.monotonic() - start
                line = f"  {icon} [{elapsed_so_far:5.1f}s] [bold]{update.source}[/bold] {update.action}: [dim]{update.detail}[/dim]"
                log_lines.append(line)

            # Rotate Hindi message every 4 seconds
            if time.monotonic() - hindi_timer > 4.0:
                hindi_msg = status_tracker.next_hindi()
                hindi_timer = time.monotonic()

            # Build display — show last 15 log lines
            visible = log_lines[-15:]
            elapsed_now = time.monotonic() - start
            display_text = f"  [bold yellow]{hindi_msg}[/bold yellow]\n\n" + "\n".join(visible)

            border = "red" if cancel_event.is_set() else "cyan"
            live.update(
                Panel(
                    display_text,
                    title="[bold cyan]⚡ Cool Code[/bold cyan]",
                    border_style=border,
                    subtitle=f"[dim]${swarm.cost_tracker.session_total:.4f} | {elapsed_now:.1f}s | ESC=cancel | type+Enter=add info[/dim]",
                    padding=(1, 2),
                )
            )

    # Signal keyboard thread to stop
    cancel_event.set()

    # Flush any pending terminal escape sequences (CPR responses like \x1b[63;1R)
    # that Rich Live's cursor position queries may have left in stdin
    _flush_stdin()

    # Handle cancellation
    if swarm._cancelled and not swarm_task.done():
        swarm_task.cancel()
        try:
            await swarm_task
        except asyncio.CancelledError:
            pass
        console.print()
        console.print("[yellow]Execution cancelled by user (ESC)[/yellow]")
        if log_lines:
            console.print(Panel(
                "\n".join(log_lines),
                title="[dim]Partial Activity Log[/dim]",
                border_style="yellow",
                padding=(1, 2),
            ))
        return

    result = swarm_task.result()
    elapsed = time.monotonic() - start

    # Show final activity log
    console.print()
    if log_lines:
        console.print(Panel(
            "\n".join(log_lines),
            title="[dim]Activity Log[/dim]",
            border_style="dim",
            padding=(1, 2),
        ))

    console.print()
    console.print(Panel(Markdown(result.output), title="[bold green]Result[/bold green]", border_style="green", padding=(1, 2)))
    console.print()
    console.print(f"[dim]Completed in {elapsed:.1f}s[/dim]")
    _print_stats(result)

    swarm.task_router.save()


async def _run_auto_pipeline(task: str, swarm: Swarm) -> None:
    """Run an autonomous pipeline with planning, gates, and checkpoints."""
    from coolcode.auto.pipeline import AutoPipeline

    pipeline = AutoPipeline(swarm=swarm, gate_frequency=2)

    # Plan
    console.print(f"\n[bold cyan]Planning autonomous pipeline...[/bold cyan]")
    plan = pipeline.plan(task)
    pipeline.show_plan(plan)

    # Confirm
    try:
        confirm = console.input("[bold]Start pipeline? (y/n): [/bold]").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]Cancelled.[/dim]")
        return
    if confirm not in ("y", "yes"):
        console.print("[dim]Pipeline cancelled.[/dim]")
        return

    console.print()

    # Execute
    result = await pipeline.execute(plan)

    # Show results
    console.print()
    if result.cancelled:
        console.print(f"[yellow]Pipeline stopped after stage {result.stages_completed}/{result.total_stages}[/yellow]")
    elif result.rollback_stage >= 0:
        console.print(f"[blue]Pipeline rolled back to before stage {result.rollback_stage + 1}[/blue]")
    else:
        console.print(f"[green]Pipeline complete: {result.stages_completed}/{result.total_stages} stages[/green]")

    # Show stage summary
    table = Table(title="Pipeline Results", show_header=True, border_style="cyan")
    table.add_column("#", width=4)
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Cost")
    table.add_column("Time")

    for sr in result.stage_results:
        status = "[green]OK[/green]" if sr.success else "[red]FAIL[/red]"
        table.add_row(
            str(sr.stage_index + 1),
            sr.description,
            status,
            f"${sr.cost_usd:.4f}",
            f"{sr.elapsed_ms / 1000:.1f}s",
        )

    console.print(table)
    console.print(f"\n[bold]Total cost:[/bold] ${result.total_cost:.4f}")

    # Show final output
    if result.final_output:
        from rich.markdown import Markdown
        console.print(Panel(
            Markdown(result.final_output[:5000]),
            title="[bold green]Final Output[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def _interactive_loop(config: CoolCodeConfig, strategy: str) -> None:
    """Run the interactive REPL with persistent swarm (session memory)."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    history_path = os.path.expanduser("~/.coolcode/history")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    session = PromptSession(history=FileHistory(history_path))

    _print_banner()

    # If no providers configured, run onboarding
    if not config.providers:
        config = _onboarding(config)

    # Create a persistent swarm for the entire session — conversation history lives here
    swarm: Swarm | None = None
    if config.providers:
        swarm = _create_swarm(config, strategy)

    # Show active state
    active = [p.provider for p in config.providers if p.api_key]
    if len(active) > 1:
        mode = "parallel racing"
    elif active:
        mode = active[0]
    else:
        mode = "none"
    console.print(f"[dim]Active: {', '.join(active)} ({mode}) | Strategy: {strategy}[/dim]")
    console.print("[dim]Commands: /model, /goal, /stats, /help, quit[/dim]")
    console.print()

    # Daemon client for checking insights
    daemon_client = None
    try:
        from coolcode.daemon.ipc import DaemonClient
        daemon_client = DaemonClient()
    except ImportError:
        pass

    while True:
        # Show daemon insights if any
        if daemon_client and daemon_client.is_running():
            insights = daemon_client.get_insights()
            for insight in insights:
                icon = {"warning": "[yellow]![/yellow]", "suggestion": "[cyan]>[/cyan]"}.get(
                    insight.severity, "[dim]*[/dim]"
                )
                console.print(f"  {icon} [dim][daemon][/dim] {insight.message}")

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
            # Recreate swarm with new provider config (but preserve conversation history + cost)
            old_history = swarm._conversation_history if swarm else []
            old_cost = swarm.cost_tracker if swarm else None
            swarm = _create_swarm(config, strategy, cost_tracker=old_cost)
            swarm._conversation_history = old_history
            continue

        if user_input.startswith("/goal"):
            from coolcode.agent.goals import list_goals, get_goal
            args = user_input[5:].strip()
            if not args:
                # Show current goal + available goals
                current = swarm.goal if swarm else "general"
                console.print(f"[bold]Current goal:[/bold] [cyan]{current}[/cyan]\n")
                console.print("[bold]Available goals:[/bold]")
                for g in list_goals():
                    marker = " [green]← active[/green]" if g["name"] == current else ""
                    console.print(f"  [cyan]{g['name']:20s}[/cyan] {g['description']}{marker}")
                console.print()
                console.print("[dim]Usage: /goal <name>[/dim]")
            else:
                # Switch goal
                valid_names = [g["name"] for g in list_goals()]
                if args not in valid_names:
                    console.print(f"[red]Unknown goal: {args}[/red]")
                    console.print(f"[dim]Available: {', '.join(valid_names)}[/dim]")
                else:
                    if swarm:
                        swarm.set_goal(args)
                    if args == "general":
                        console.print("[green]Switched to general mode — queen routes everything[/green]")
                    else:
                        pipeline = get_goal(args)
                        console.print(f"[green]Switched to goal: {args}[/green]")
                        console.print(f"[dim]{pipeline.description}[/dim]")
                        console.print(f"[dim]Pipeline: {len(pipeline.stages)} stages[/dim]")
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

            # Cost stats
            if swarm:
                ct = swarm.cost_tracker
                inp, out = ct.session_tokens
                console.print(f"\n[bold]Cost:[/bold]")
                console.print(f"  Session: ${ct.session_total:.4f}")
                console.print(f"  Today: ${ct.daily_total():.4f}")
                console.print(f"  Tokens: {inp:,} in / {out:,} out")
                breakdown = ct.provider_breakdown()
                for key, info in breakdown.items():
                    console.print(f"    {key}: ${info['cost_usd']:.4f} ({info['calls']} calls)")
                budget = ct.check_budget(
                    daily_limit=config.cost.budget_daily_usd,
                    session_limit=config.cost.budget_session_usd,
                )
                if not budget.ok:
                    console.print(f"  [yellow]{budget}[/yellow]")

            # Memory stats
            if swarm:
                kg_stats = swarm.knowledge_graph.stats
                console.print(f"\n[bold]Knowledge Graph:[/bold]")
                console.print(f"  Nodes: {kg_stats['nodes']}, Edges: {kg_stats['edges']}")
                console.print(f"\n[bold]Session:[/bold]")
                console.print(f"  Conversation turns: {len(swarm._conversation_history)}")
            continue

        if user_input.startswith("/budget"):
            args = user_input[7:].strip()
            if not args:
                console.print(f"[bold]Budget:[/bold]")
                console.print(f"  Daily limit:   ${config.cost.budget_daily_usd:.2f}" if config.cost.budget_daily_usd > 0 else "  Daily limit:   unlimited")
                console.print(f"  Session limit: ${config.cost.budget_session_usd:.2f}" if config.cost.budget_session_usd > 0 else "  Session limit: unlimited")
                if swarm:
                    console.print(f"  Session spent: ${swarm.cost_tracker.session_total:.4f}")
                    console.print(f"  Today spent:   ${swarm.cost_tracker.daily_total():.4f}")
                console.print("[dim]Usage: /budget daily 5.00 | /budget session 2.00 | /budget off[/dim]")
            elif args == "off":
                config.cost.budget_daily_usd = 0.0
                config.cost.budget_session_usd = 0.0
                console.print("[green]Budget limits removed[/green]")
            elif args.startswith("daily"):
                try:
                    config.cost.budget_daily_usd = float(args.split()[1])
                    console.print(f"[green]Daily budget set to ${config.cost.budget_daily_usd:.2f}[/green]")
                except (IndexError, ValueError):
                    console.print("[red]Usage: /budget daily <amount>[/red]")
            elif args.startswith("session"):
                try:
                    config.cost.budget_session_usd = float(args.split()[1])
                    console.print(f"[green]Session budget set to ${config.cost.budget_session_usd:.2f}[/green]")
                except (IndexError, ValueError):
                    console.print("[red]Usage: /budget session <amount>[/red]")
            continue

        if user_input == "/cost":
            if swarm:
                ct = swarm.cost_tracker
                inp, out = ct.session_tokens
                console.print(f"[bold]Session:[/bold] ${ct.session_total:.4f} ({inp:,} in / {out:,} out)")
                console.print(f"[bold]Today:[/bold]   ${ct.daily_total():.4f}")
            else:
                console.print("[dim]No cost data yet[/dim]")
            continue

        if user_input.startswith("/daemon"):
            from coolcode.daemon.ipc import DaemonClient
            from coolcode.daemon.server import daemonize, is_daemon_running, stop_daemon, get_daemon_pid
            args = user_input[7:].strip()
            if args == "start":
                if is_daemon_running():
                    console.print(f"[yellow]Daemon already running (PID: {get_daemon_pid()})[/yellow]")
                else:
                    project_dir = config.project_dir
                    daemonize(project_dir)
                    console.print(f"[green]Daemon started watching: {project_dir}[/green]")
            elif args == "stop":
                if stop_daemon():
                    console.print("[green]Daemon stopped[/green]")
                else:
                    console.print("[yellow]No daemon running[/yellow]")
            elif args == "status":
                client = DaemonClient()
                if client.is_running():
                    status = client.get_status()
                    if status:
                        console.print(f"[green]Daemon running[/green]")
                        console.print(f"  Project: {status.get('project_dir', '?')}")
                        console.print(f"  Uptime: {status.get('uptime_s', 0):.0f}s")
                        stats = status.get("stats", {})
                        console.print(f"  Events tracked: {stats.get('total_events', 0)}")
                        console.print(f"  Unique files: {stats.get('unique_files', 0)}")
                    else:
                        console.print(f"[green]Daemon running (PID: {get_daemon_pid()})[/green]")
                else:
                    console.print("[dim]Daemon not running. Use /daemon start[/dim]")
            else:
                console.print("[bold]Daemon Mode[/bold] — background watcher + proactive insights")
                console.print("  /daemon start    — Start watching this project")
                console.print("  /daemon stop     — Stop the daemon")
                console.print("  /daemon status   — Show daemon status")
            continue

        if user_input == "/help":
            console.print("[bold]Commands:[/bold]")
            console.print("  /model              — Show active models")
            console.print("  /model setup        — Re-run provider setup")
            console.print("  /model claude       — Switch to Claude only")
            console.print("  /model minimax      — Switch to MiniMax only")
            console.print("  /model both         — Both providers (parallel racing)")
            console.print("  /model claude:claude-opus-4-6  — Switch specific model")
            console.print("  /goal               — Show current goal + available goals")
            console.print("  /goal code-review   — Deep code review mode")
            console.print("  /goal cyber-security — Security audit (OWASP/CWE/SANS)")
            console.print("  /goal build-feature — Plan → implement → test → review")
            console.print("  /goal debug         — Diagnose → fix → verify")
            console.print("  /goal explain       — Architecture explanation")
            console.print("  /goal optimize      — Performance optimization")
            console.print("  /goal general       — Back to default (queen routes)")
            console.print("  /auto <task>        — Autonomous pipeline with gates & checkpoints")
            console.print("  /cost               — Quick cost summary")
            console.print("  /budget             — View/set budget caps")
            console.print("  /budget daily 5.00  — Set $5/day limit")
            console.print("  /budget session 2   — Set $2/session limit")
            console.print("  /budget off         — Remove budget limits")
            console.print("  /daemon start       — Start background watcher")
            console.print("  /daemon stop        — Stop background watcher")
            console.print("  /daemon status      — Show daemon info")
            console.print("  /stats              — Provider stats, learnings, cost, and memory")
            console.print("  /help               — Show this help")
            console.print("  quit                — Exit Cool Code")
            continue

        if user_input.startswith("/auto"):
            args = user_input[5:].strip()
            if not args:
                console.print("[bold]Autonomous Pipeline[/bold]")
                console.print("  Run multi-stage tasks with checkpoints and human approval gates.")
                console.print()
                console.print("[dim]Usage: /auto <task description>[/dim]")
                console.print("[dim]  /auto build user dashboard with auth and tests[/dim]")
                console.print("[dim]  /auto refactor the payment module end to end[/dim]")
                continue
            if not swarm:
                if not config.providers:
                    console.print("[red]No providers configured. Run /model setup[/red]")
                    continue
                swarm = _create_swarm(config, strategy)
            try:
                asyncio.run(_run_auto_pipeline(args, swarm))
            except Exception as e:
                console.print(f"[red]Auto pipeline error: {e}[/red]")
                logger.exception("Auto pipeline failed")
            console.print()
            continue

        if not config.providers:
            console.print("[red]No providers configured. Run /model setup[/red]")
            continue

        if not swarm:
            swarm = _create_swarm(config, strategy)

        try:
            asyncio.run(_run_task(user_input, swarm))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.exception("Task execution failed")

        console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _ensure_path() -> None:
    """Ensure the coolcode script directory is in PATH. Auto-fix if possible."""
    import shutil
    import subprocess

    # If `coolcode` is already findable via PATH, nothing to do
    if shutil.which("coolcode"):
        return

    # Find where pip installed the script
    user_bin = Path(sys.executable).parent  # e.g. /usr/local/bin or ~/.local/bin
    # Also check the user-scheme bin
    try:
        import site
        user_base = Path(site.getusersitepackages()).parent  # e.g. ~/Library/Python/3.11
        user_script_dir = user_base / "bin"
    except Exception:
        user_script_dir = None

    script_path = None
    for candidate in [user_bin / "coolcode", user_script_dir and user_script_dir / "coolcode"]:
        if candidate and candidate.exists():
            script_path = candidate
            break

    if not script_path:
        return  # Can't find the script, nothing to fix

    script_dir = str(script_path.parent)

    # Check if script_dir is already in PATH
    if script_dir in os.environ.get("PATH", "").split(":"):
        return

    # Try to create a symlink in /usr/local/bin (always in PATH on macOS/Linux)
    symlink_target = Path("/usr/local/bin/coolcode")
    if symlink_target.parent.exists() and not symlink_target.exists():
        try:
            symlink_target.symlink_to(script_path)
            console.print(f"[green]✓[/green] [dim]Linked coolcode → /usr/local/bin/coolcode[/dim]")
            return
        except PermissionError:
            # Try with sudo
            try:
                subprocess.run(
                    ["sudo", "ln", "-sf", str(script_path), str(symlink_target)],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                console.print(f"[green]✓[/green] [dim]Linked coolcode → /usr/local/bin/coolcode[/dim]")
                return
            except Exception:
                pass

    # Fallback: add to shell profile
    shell = os.environ.get("SHELL", "")
    rc_file = Path.home() / (".zshrc" if "zsh" in shell else ".bashrc")
    export_line = f'export PATH="{script_dir}:$PATH"'

    # Check if already in rc file
    if rc_file.exists() and export_line in rc_file.read_text():
        console.print(f"[yellow]PATH is set in {rc_file.name} but not loaded. Run:[/yellow]")
        console.print(f"  [bold]source ~/{rc_file.name}[/bold]")
        return

    # Add to rc file
    try:
        with open(rc_file, "a") as f:
            f.write(f"\n# Cool Code CLI\n{export_line}\n")
        console.print(f"[green]✓[/green] [dim]Added coolcode to PATH in ~/{rc_file.name}[/dim]")
        console.print(f"  [dim]Run [bold]source ~/{rc_file.name}[/bold] or open a new terminal[/dim]")
    except Exception:
        console.print(f"[yellow]Add this to your ~/{rc_file.name}:[/yellow]")
        console.print(f"  [bold]{export_line}[/bold]")


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

    # Auto-fix PATH on first run so `coolcode` works in future sessions
    _ensure_path()

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
            swarm = _create_swarm(config, strategy)
            asyncio.run(_run_task(task, swarm))
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
