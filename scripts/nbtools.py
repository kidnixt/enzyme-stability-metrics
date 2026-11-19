"""Minimal helper to emit valid .ipynb files.

The one thing that is easy to get wrong: in the notebook format, `source` is a
list of strings and each one must KEEP its trailing newline. Splitting on "\n"
without re-adding them makes Jupyter render the whole cell on a single line.
"""
import json


def _lines(text: str):
    text = text.strip("\n")
    return [ln + "\n" for ln in text.split("\n")[:-1]] + [text.split("\n")[-1]]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(text)}


def write(cells, path: str, kernel_name: str = "python3",
          display_name: str = "Python 3") -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": display_name, "language": "python",
                           "name": kernel_name},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
