"""Every nontrivial number in the paper must appear in a committed source.

This is a transcription guard: it catches typos and numbers with no recorded
source. It does not establish that a source's own measurement is correct.
"""
from __future__ import annotations
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "nla_three_families.md"
FIGURES = ROOT / "paper" / "make_figures.py"
NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
URL = re.compile(r"<?https?://[^\s>)]+>?")
YEAR = re.compile(r"(?:19|20)\d\d")
# Section numbering is structure, not data: "### 4.3 Title", "§4.5–4.6", "Section 4.3".
SECTION = re.compile(r"^#+\s+\d+(?:\.\d+)*|§\d+(?:\.\d+)*(?:–\d+(?:\.\d+)*)?|Section \d+(?:\.\d+)*",
                     re.M)


def declared_sources(text: str) -> list[Path]:
    header = re.search(r"<!--(.*?)-->", text, re.S).group(1)
    listing = header.split("sources:", 1)[1]
    return [ROOT / line.strip() for line in listing.splitlines() if line.strip()]


def numbers(text: str) -> set[str]:
    return set(NUMBER.findall(text))


def nontrivial(token: str) -> bool:
    """Decimals and numbers of three or more digits; publication years excluded."""
    if YEAR.fullmatch(token):
        return False
    return "." in token or len(token.replace(",", "")) >= 3


def untraced(text: str, corpus: set[str]) -> list[str]:
    body = SECTION.sub(" ", URL.sub(" ", re.sub(r"<!--.*?-->", " ", text, flags=re.S)))
    return sorted(token for token in numbers(body) if nontrivial(token) and token not in corpus)


def figure_values() -> list[str]:
    tree = ast.parse(FIGURES.read_text("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "UPPER_BOUNDS":
            rows = ast.literal_eval(node.value)
            return [repr(value) for _, *values in rows for value in values]
    raise AssertionError("UPPER_BOUNDS not found in make_figures.py")


def corpus() -> set[str]:
    text = PAPER.read_text("utf-8")
    return set().union(*(numbers(path.read_text("utf-8")) for path in declared_sources(text)))


def test_declared_sources_exist():
    missing = [str(p) for p in declared_sources(PAPER.read_text("utf-8")) if not p.is_file()]
    assert not missing, f"paper cites missing sources: {missing}"


def test_every_paper_number_is_in_a_declared_source():
    assert untraced(PAPER.read_text("utf-8"), corpus()) == []


def test_every_figure_value_is_in_a_declared_source():
    known = corpus()
    assert [v for v in figure_values() if v not in known] == []


def test_checker_flags_an_unrecorded_number():
    known = {"0.8105", "41.4"}
    assert untraced("accuracy 0.8105, loss 41.4, invented 0.12345", known) == ["0.12345"]


def test_checker_ignores_years_urls_sections_and_small_integers():
    text = ("### 4.3 Title\nZhang and Nanda (2023) <https://arxiv.org/abs/2309.16042>, "
            "3 of 8 steps; see §4.5–4.6 and Section 4.3")
    assert untraced(text, set()) == []


def test_section_stripping_keeps_data_on_the_same_line():
    assert untraced("### 4.3 Loss rose to 41.41 pp", set()) == ["41.41"]
