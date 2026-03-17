"""Code search tools — grep and definition finding."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def grep_search(
    pattern: str,
    directory: str = ".",
    file_glob: str = "",
    max_results: int = 50,
    case_insensitive: bool = False,
) -> str:
    """Search file contents using regex pattern (ripgrep if available, else fallback).

    Args:
        pattern: Regex pattern to search for.
        directory: Root directory to search in.
        file_glob: Optional glob to filter files (e.g., '*.py').
        max_results: Maximum number of results to return.
        case_insensitive: Whether to ignore case.

    Returns:
        Matching lines with file paths and line numbers.
    """
    cwd = Path(directory).resolve()
    if not cwd.exists():
        return f"Error: Directory not found: {cwd}"

    # Try ripgrep first (faster)
    cmd = ["rg", "--line-number", "--no-heading", f"--max-count={max_results}"]
    if case_insensitive:
        cmd.append("-i")
    if file_glob:
        cmd.extend(["--glob", file_glob])
    cmd.append(pattern)
    cmd.append(str(cwd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode <= 1:  # 0=matches, 1=no matches
            lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
            if len(lines) > max_results:
                lines = lines[:max_results]
                lines.append(f"... (limited to {max_results} results)")
            return "\n".join(lines) if lines else f"No matches for pattern: {pattern}"
    except FileNotFoundError:
        pass  # ripgrep not installed, fallback below
    except subprocess.TimeoutExpired:
        return "Error: Search timed out"

    # Fallback: Python-based search
    try:
        regex = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
    except re.error as e:
        return f"Error: Invalid regex: {e}"

    results: list[str] = []
    glob_pattern = file_glob or "**/*"
    for filepath in cwd.glob(glob_pattern):
        if not filepath.is_file():
            continue
        if filepath.name.startswith("."):
            continue
        try:
            text = filepath.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    rel = filepath.relative_to(cwd)
                    results.append(f"{rel}:{i}: {line.rstrip()}")
                    if len(results) >= max_results:
                        results.append(f"... (limited to {max_results} results)")
                        return "\n".join(results)
        except (OSError, UnicodeDecodeError):
            continue

    return "\n".join(results) if results else f"No matches for pattern: {pattern}"


def find_definition(
    name: str,
    directory: str = ".",
    language: str = "",
) -> str:
    """Find the definition of a class, function, or variable in the codebase.

    Args:
        name: The name to search for (class, function, variable).
        directory: Root directory to search in.
        language: Optional language hint ('python', 'typescript', 'go', etc.)

    Returns:
        File paths and lines where the definition is found.
    """
    patterns: dict[str, list[str]] = {
        "python": [
            rf"^\s*(def|class|async\s+def)\s+{re.escape(name)}\b",
            rf"^{re.escape(name)}\s*=",
        ],
        "typescript": [
            rf"^\s*(export\s+)?(function|class|const|let|var|interface|type|enum)\s+{re.escape(name)}\b",
        ],
        "javascript": [
            rf"^\s*(export\s+)?(function|class|const|let|var)\s+{re.escape(name)}\b",
        ],
        "go": [
            rf"^\s*(func|type|var|const)\s+(\([^)]*\)\s+)?{re.escape(name)}\b",
        ],
        "rust": [
            rf"^\s*(pub\s+)?(fn|struct|enum|trait|type|const|static)\s+{re.escape(name)}\b",
        ],
    }

    if language and language.lower() in patterns:
        search_patterns = patterns[language.lower()]
        ext_map = {
            "python": "*.py", "typescript": "*.ts", "javascript": "*.js",
            "go": "*.go", "rust": "*.rs",
        }
        file_glob = ext_map.get(language.lower(), "")
    else:
        # Try all patterns
        search_patterns = [p for plist in patterns.values() for p in plist]
        file_glob = ""

    all_results: list[str] = []
    for pat in search_patterns:
        result = grep_search(pat, directory, file_glob=file_glob, max_results=10)
        if result and not result.startswith("No matches"):
            all_results.append(result)

    return "\n".join(all_results) if all_results else f"No definition found for: {name}"
