"""Built-in tools for Cool Code agents."""

from coolcode.tools.files import read_file, write_file, edit_file, list_dir, glob_search
from coolcode.tools.shell import execute_shell
from coolcode.tools.search import grep_search, find_definition
from coolcode.tools.git import git_status, git_diff, git_log, git_commit

ALL_TOOLS = [
    read_file,
    write_file,
    edit_file,
    list_dir,
    glob_search,
    execute_shell,
    grep_search,
    find_definition,
    git_status,
    git_diff,
    git_log,
    git_commit,
]

__all__ = ["ALL_TOOLS"]
