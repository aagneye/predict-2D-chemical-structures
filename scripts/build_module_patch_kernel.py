"""Assemble a kernel from a public base notebook with one or more pinned modules replaced.

The 0.477 lineage embeds its helper modules as ``MODULES['name.py'] = '...'`` lines and
checks each against ``MODULE_SHA256['name.py']`` before running it. ``--module
name.py=path`` replaces that line with our file and updates the pin, so the base's own
integrity check passes for the new code and still guards every other module. All other
cells, the pinned image and the inputs stay as in the base; ``--extra-dataset`` adds inputs.

Example:
    python scripts/build_module_patch_kernel.py --base-dir ~/forks/ours_0477 \\
        --module neural_scores.py=kaggle/neural_ensemble/neural_scores.py \\
        --extra-dataset aagneye/casmi26-neural-seeds --out ~/forks/neural_ensemble \\
        --id aagneye/casmi26-lb0477-neural-seed-ensemble \\
        --title "CASMI26 LB 0.477 neural seed ensemble"
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def patch_module(nb: dict, name: str, source: str) -> tuple[dict, str, str]:
    """Return ``(notebook, old_sha, new_sha)`` with ``MODULES[name]`` set to ``source`` and
    its ``MODULE_SHA256`` pin updated. Raises if the module line or the pin is not found
    exactly once."""
    prefix = f"MODULES[{name!r}] = "
    old = None
    cells = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            cells.append(cell)
            continue
        lines = "".join(cell["source"]).split("\n")
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                if old is not None:
                    raise ValueError(f"{name}: module defined twice")
                old = ast.literal_eval(line[len(prefix):])
                lines[i] = prefix + repr(source)
        cells.append(dict(cell, source=["\n".join(lines)]))
    if old is None:
        raise ValueError(f"{name}: no MODULES line")
    old_sha, new_sha = sha256(old), sha256(source)
    pinned = [i for i, c in enumerate(cells) if c["cell_type"] == "code"
              and old_sha in "".join(c["source"])]
    if len(pinned) != 1 or "".join(cells[pinned[0]]["source"]).count(old_sha) != 1:
        raise ValueError(f"{name}: expected one MODULE_SHA256 pin {old_sha[:12]}")
    i = pinned[0]
    cells[i] = dict(cells[i], source=["".join(cells[i]["source"]).replace(old_sha, new_sha)])
    return dict(nb, cells=cells), old_sha, new_sha


def kernel_metadata(base_meta: dict, kernel_id: str, title: str,
                    extra_datasets: list[str]) -> dict:
    meta = dict(base_meta)
    meta.pop("id_no", None)
    meta.update(id=kernel_id, title=title, code_file=f"{kernel_id.split('/')[1]}.ipynb",
                is_private=True)
    datasets = list(base_meta.get("dataset_sources", []))
    datasets += [d for d in extra_datasets if d not in datasets]
    meta["dataset_sources"] = datasets
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-dir", required=True, help="folder from `kaggle kernels pull -m`")
    parser.add_argument("--module", action="append", required=True, help="name.py=path")
    parser.add_argument("--extra-dataset", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--id", required=True, help="owner/slug of the new kernel")
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    base_meta = json.loads((base_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    nb = json.loads((base_dir / base_meta["code_file"]).read_text(encoding="utf-8"))
    changes = []
    for spec in args.module:
        name, path = spec.split("=", 1)
        nb, old_sha, new_sha = patch_module(nb, name, Path(path).read_text(encoding="utf-8"))
        changes.append(f"`{name}` {old_sha[:12]} → {new_sha[:12]}")
    nb["cells"] = [{"cell_type": "markdown", "metadata": {}, "source": [
        f"# Fork of [{base_meta['id']}](https://www.kaggle.com/code/{base_meta['id']})\n\n"
        "Unchanged except for these pinned modules (the notebook's own SHA-256 check covers "
        "them): " + "; ".join(changes) + ". All upstream credits are retained; see "
        "github.com/aagneye/predict-2D-chemical-structures."]}] + nb["cells"]
    meta = kernel_metadata(base_meta, args.id, args.title, args.extra_dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / meta["code_file"]).write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                                         encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] {out / meta['code_file']}: " + "; ".join(changes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
