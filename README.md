# Code-Cube-COMP-5004-Assignment-2.1-Programming-Task-1

NumCompute — a modular scientific computing toolkit built on plain Python and NumPy.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The only runtime dependency is NumPy; `pytest` is the only development dependency.
