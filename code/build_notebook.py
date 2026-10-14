"""Compile a `# %%`-marked Python source file into a Jupyter .ipynb.

Cell markers:
    # %% [markdown]   -> markdown cell (body lines must start with '# ')
    # %%              -> code cell
"""
import ast
import json
import sys
from pathlib import Path


def split_cells(text):
    cells, cur_kind, cur_lines = [], None, []

    def flush():
        if cur_kind is None:
            return
        body = "\n".join(cur_lines).strip("\n")
        if body.strip():
            cells.append((cur_kind, body))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            cur_kind = "markdown" if "[markdown]" in stripped else "code"
            cur_lines = []
        else:
            cur_lines.append(line)
    flush()
    return cells


def demote_markdown(body):
    out = []
    for line in body.splitlines():
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out).strip("\n")


def as_source(body):
    lines = body.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main(src_path, out_path):
    text = Path(src_path).read_text(encoding="utf-8")
    cells_raw = split_cells(text)

    nb_cells, n_code, n_md, errors = [], 0, 0, []
    for idx, (kind, body) in enumerate(cells_raw):
        if kind == "markdown":
            n_md += 1
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": as_source(demote_markdown(body)),
            })
        else:
            n_code += 1
            try:
                ast.parse(body)
            except SyntaxError as exc:
                errors.append(f"cell #{idx} (code #{n_code}): {exc}")
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": as_source(body),
            })

    if errors:
        print("SYNTAX ERRORS:", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1

    nb = {
        "cells": nb_cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(out_path).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK  {out_path}  cells={len(nb_cells)} (code={n_code}, markdown={n_md})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
