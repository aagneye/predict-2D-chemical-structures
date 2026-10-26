"""Assemble the overlay kernel: a public base notebook, unchanged, plus our overlay cell.

The base (a public Kaggle notebook, Apache 2.0) is pulled with ``kaggle kernels pull -m``
into ``--base-dir``; its cells, pinned Docker image and inputs are kept exactly. We prepend
an attribution cell and append ``kaggle/overlay/overlay_cell.py`` as the last code cell,
which runs our pipeline after the base wrote ``submission.csv`` and inserts our derived
structures (see ``casmi.overlay``). ``--base-sha256`` pins the base notebook file.

Example:
    python scripts/build_overlay_kernel.py --base-dir ~/forks/alexchilton_casmi26-lb-0-467 \\
        --base-sha256 <hex> --out ~/forks/overlay_kernel \\
        --id aagneye/casmi26-lb0467-class3-overlay --title "CASMI26 0.467 + class-3 overlay" \\
        --extra-dataset aagneye/casmi26-baseline-assets \\
        --extra-dataset ahmedberatozer/casmi26-pubchem-tier
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

OVERLAY_CELL = Path(__file__).resolve().parents[1] / "kaggle" / "overlay" / "overlay_cell.py"


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code_cell(code: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [code]}


def build(
    base_nb: dict,
    base_meta: dict,
    overlay_code: str,
    kernel_id: str,
    title: str,
    extra_datasets: list[str],
    attribution: str,
) -> tuple[dict, dict]:
    """Return ``(notebook, kernel_metadata)`` for the overlay kernel.

    The base's cells are kept in order and unmodified; ``kernel_sources`` and the pinned
    image come from the base; ``dataset_sources`` are the base's plus ``extra_datasets``.
    """
    nb = dict(base_nb)
    nb["cells"] = (
        [markdown_cell(attribution)]
        + list(base_nb["cells"])
        + [markdown_cell("## Class-3 overlay (ours)\n\nInserts one analog-derived structure "
                         "below the base's rank 1 for molecules without a close library "
                         "match. Base rank 1 and candidate set are otherwise unchanged; "
                         "`submission_base_before_overlay.csv` keeps the base's output."),
           code_cell(overlay_code)]
    )
    meta = dict(base_meta)
    meta.pop("id_no", None)
    slug = kernel_id.split("/")[1]
    meta.update(id=kernel_id, title=title, code_file=f"{slug}.ipynb", is_private=True)
    datasets = list(base_meta.get("dataset_sources", []))
    for name in extra_datasets:
        if name not in datasets:
            datasets.append(name)
    meta["dataset_sources"] = datasets
    return nb, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, help="folder from `kaggle kernels pull -m`")
    parser.add_argument("--base-sha256", default=None, help="expected sha256 of the base .ipynb")
    parser.add_argument("--out", required=True)
    parser.add_argument("--id", required=True, help="owner/slug of the new kernel")
    parser.add_argument("--title", required=True)
    parser.add_argument("--extra-dataset", action="append", default=[])
    parser.add_argument("--overlay", default=str(OVERLAY_CELL))
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    base_meta = json.loads((base_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    base_path = base_dir / base_meta["code_file"]
    digest = hashlib.sha256(base_path.read_bytes()).hexdigest()
    if args.base_sha256 and digest != args.base_sha256:
        raise SystemExit(f"base notebook sha256 {digest} != expected {args.base_sha256}")
    base_nb = json.loads(base_path.read_text(encoding="utf-8"))
    attribution = (
        f"# Overlay on [{base_meta['id']}](https://www.kaggle.com/code/{base_meta['id']})\n\n"
        f"The base notebook (Apache 2.0, sha256 `{digest}`) runs unchanged below with its "
        "own inputs and pinned image; all upstream credits are retained. Our cell at the "
        "end adds structures that are in no database (class 3) — see "
        "github.com/aagneye/predict-2D-chemical-structures."
    )
    nb, meta = build(base_nb, base_meta, Path(args.overlay).read_text(encoding="utf-8"),
                     args.id, args.title, args.extra_dataset, attribution)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / meta["code_file"]).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] {out / meta['code_file']}: {len(nb['cells'])} cells, "
          f"{len(meta['dataset_sources'])} datasets, base sha256 {digest[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
