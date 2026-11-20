"""Small helpers for what is safe to show a user."""
import re

_PATH = re.compile(r"(?:[A-Za-z]:)?[\\/][^\s'\"]*[\\/][^\s'\"]+")
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def friendly(e, limit=600) -> str:
    """An error message for the user: trimmed, with any server file paths removed. A missing ffmpeg, which Windows
    reports only as "the system cannot find the file specified", is said in plain words with what to do."""
    from . import tools
    msg = _ANSI.sub("", str(e))
    msg = re.sub(r"^(?:ERROR:\s*)+", "", msg.strip())
    return tools.explain(e) or _PATH.sub("[file]", msg[-limit:])


