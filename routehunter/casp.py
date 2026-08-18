from dataclasses import dataclass

LINK_PLACEHOLDER = "Cached predicted routes are not available yet"


@dataclass
class CaspSolvedEntry:
    tool: str
    tool_display: str
    link: str
