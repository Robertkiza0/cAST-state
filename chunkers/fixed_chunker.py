"""Baseline 1 (`fixed`): fixed-size / line-based chunking, no AST awareness at
all — the classic baseline the cAST paper compares against (Figure 1/2:
"syntax-agnostic chunking often omits crucial information").

Uses the SAME size metric as cast_orig/cast_scope (non-whitespace character
count) and the same max_chunk_size budget, so all three baselines are
compared under an identical size control, not just an identical chunker name
(the cAST paper itself only accepts this comparison as fair once chunk-size
statistics line up — see its Table 4 / Section 4 "Selection of maximum chunk
size"). Splits only ever happen on line boundaries — never mid-line — same as
astchunk_reference's own toy baseline (examples/fixed_chunking.py), just
budgeted in characters instead of lines.
"""


def _nws_count(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def fixed_size_chunk(code: str, max_chunk_size: int) -> list[str]:
    """Split code into consecutive, line-aligned chunks of up to
    max_chunk_size non-whitespace characters each."""
    if not code:
        return []

    lines = code.splitlines(keepends=True)
    chunks: list[str] = []
    current_lines: list[str] = []
    current_size = 0

    for line in lines:
        line_size = _nws_count(line)
        if current_lines and current_size + line_size > max_chunk_size:
            chunks.append("".join(current_lines))
            current_lines = []
            current_size = 0
        current_lines.append(line)
        current_size += line_size

    if current_lines:
        chunks.append("".join(current_lines))

    return chunks
