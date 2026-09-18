"""Enveda CASMI 2026 — predict 2D chemical structures from LC-MS/MS spectra.

See ``docs/06-implementation-plan.md`` for the build plan this package
implements.

Import layout is deliberately shallow: heavy optional dependencies (torch)
live in :mod:`casmi.models` and are *not* imported here, so that the baseline
pipeline works in a core-only install.
"""

__version__ = "0.1.0"
