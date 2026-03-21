"""Human gates for autonomous pipelines.

Gates pause execution between stages for human approval.
The user can approve, reject, edit instructions, or rollback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)

console = Console()


class GateDecision(str, Enum):
    """Human decision at a pipeline gate."""

    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    ROLLBACK = "rollback"


@dataclass
class GateResult:
    """Result of a human gate interaction."""

    decision: GateDecision
    edit_instructions: str = ""  # only set if decision == EDIT


class GateManager:
    """Manages human approval gates in autonomous pipelines."""

    def prompt_gate(
        self,
        stage_index: int,
        stage_label: str,
        stage_output: str,
        total_stages: int,
        cost_so_far: float = 0.0,
        files_changed: list[str] | None = None,
    ) -> GateResult:
        """Pause execution and ask the user for approval.

        Shows:
        - Stage results summary
        - Cost so far
        - Files changed
        - Options: Approve / Reject / Edit / Rollback
        """
        console.print()

        # Show stage results
        console.print(Panel(
            Markdown(stage_output[:3000]),
            title=f"[bold yellow]Gate: Stage {stage_index + 1}/{total_stages} — {stage_label}[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))

        # Show summary info
        info_table = Table(show_header=False, border_style="dim", padding=(0, 2))
        info_table.add_column("Key", style="dim")
        info_table.add_column("Value", style="bold")
        info_table.add_row("Stage", f"{stage_index + 1}/{total_stages}: {stage_label}")
        info_table.add_row("Cost so far", f"${cost_so_far:.4f}")
        if files_changed:
            info_table.add_row("Files changed", ", ".join(files_changed[:10]))
        console.print(info_table)

        # Prompt for decision
        console.print()
        console.print("[bold]What would you like to do?[/bold]")
        console.print("  [green]a[/green] — Approve and continue")
        console.print("  [red]r[/red] — Reject and stop pipeline")
        console.print("  [yellow]e[/yellow] — Edit: give corrective instructions and re-run this stage")
        console.print("  [blue]b[/blue] — Rollback to before this stage")
        console.print()

        while True:
            try:
                choice = console.input("[bold]Decision (a/r/e/b): [/bold]").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return GateResult(decision=GateDecision.REJECT)

            if choice in ("a", "approve", "y", "yes"):
                console.print("[green]Approved — continuing pipeline[/green]")
                return GateResult(decision=GateDecision.APPROVE)

            elif choice in ("r", "reject", "n", "no"):
                console.print("[red]Rejected — stopping pipeline[/red]")
                return GateResult(decision=GateDecision.REJECT)

            elif choice in ("e", "edit"):
                try:
                    instructions = console.input("[bold]Enter corrective instructions: [/bold]").strip()
                except (EOFError, KeyboardInterrupt):
                    return GateResult(decision=GateDecision.REJECT)
                if instructions:
                    console.print(f"[yellow]Re-running stage with: {instructions[:80]}[/yellow]")
                    return GateResult(decision=GateDecision.EDIT, edit_instructions=instructions)
                console.print("[dim]No instructions given, try again[/dim]")

            elif choice in ("b", "back", "rollback"):
                console.print("[blue]Rolling back to before this stage[/blue]")
                return GateResult(decision=GateDecision.ROLLBACK)

            else:
                console.print("[dim]Please enter a, r, e, or b[/dim]")
