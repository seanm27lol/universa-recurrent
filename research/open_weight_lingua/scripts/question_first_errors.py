"""Post-hoc breakdown of a Phase Six run: what are the wrong answers?

Usage:
    python scripts/question_first_errors.py RUN_DIR

Descriptive only; it can never change gate U6, G6 or E6. Example: asked
"What is x at the end of this program?" before `x = 7 / y = 16 / y = y + 1 /
y = y - 3`, a model that answers 14 has given y's final value. For each
format, this prints:
- how many wrong answers equal the other variable's final value (counted
  only when the two final values differ);
- accuracy split by whether the program's last line assigns the asked
  variable or the other one.
"""

import json
from pathlib import Path
import sys


def breakdown(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    generations = json.loads((run / "results.json").read_text())["generations"]
    prompts = {row["id"]: row["prompt"] for row in manifest["rows"]}
    final = {}
    for record in manifest["records"]:  # in line order, so the last one per prompt holds the final values
        final[record["id"]] = record
    out = {}
    for name in ("question_first", "question_after"):
        table = {"wrong": 0, "wrong_with_other_value": 0}
        for split in ("asked", "other"):
            table[f"last_line_assigns_{split}"] = {"prompts": 0, "correct": 0, "answered_other_value": 0}
        for generation in generations:
            asked = generation["variable"]
            other = "y" if asked == "x" else "x"
            last_line = prompts[generation["id"]].rsplit("\n", 1)[1]
            split = "asked" if last_line.split(" = ")[0] == asked else "other"
            text = generation[name]["text"].strip()
            values = final[generation["id"]]
            gave_other = (
                text.isdigit() and int(text) == values[other]["value"] and values[other]["value"] != values[asked]["value"]
            )
            cell = table[f"last_line_assigns_{split}"]
            cell["prompts"] += 1
            cell["correct"] += generation[name]["correct"]
            cell["answered_other_value"] += gave_other
            if not generation[name]["correct"]:
                table["wrong"] += 1
                table["wrong_with_other_value"] += gave_other
        for split in ("asked", "other"):
            cell = table[f"last_line_assigns_{split}"]
            cell["accuracy"] = cell["correct"] / cell["prompts"]
        out[name] = table
    return out


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    print(json.dumps(breakdown(Path(argv[0])), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
