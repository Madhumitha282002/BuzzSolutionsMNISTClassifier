# pytest -v error

## Error

Running `pytest -v` from the repo root failed during test collection for all three test modules:

```
ERROR collecting tests/test_data.py
ERROR collecting tests/test_evaluation.py
ERROR collecting tests/test_model.py
ModuleNotFoundError: No module named 'digit_classification'
```

## Cause

The `digit_classification` package lives under `src/digit_classification`, and the
project already has a virtual environment at `digit/` with the package installed
in editable mode (`pip show` confirmed an editable install pointing at this repo).

However, `pytest -v` was run with the system Python
(`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`) instead of the
project's venv (`digit/bin/python`). The system Python has no knowledge of the
editable install, so it can't find `digit_classification`, and `digit/bin/python`
itself didn't have `pytest` installed at all (only the base runtime deps).

## Fix

Installed the project's test dependencies into the existing venv:

```bash
./digit/bin/python -m pip install -e ".[test]"
```

This installs `pytest`/`pytest-cov` plus the project's runtime dependencies into
`digit/`, and re-registers the editable install of `digit_classification`.

## Verification

```bash
./digit/bin/python -m pytest -v
```

Result: 13 passed, 2 warnings (unrelated Lightning warnings about `self.log()`
being called before a `Trainer` is attached — not a failure).

## Going forward

Activate the venv before running tests, or invoke pytest through it directly:

```bash
source digit/bin/activate
pytest -v
```

or

```bash
./digit/bin/python -m pytest -v
```

Plain `pytest -v` will keep failing if it resolves to the system Python instead
of `digit/bin/python`.

## Update: same error even after activating the venv

After activating `digit` (prompt showed `(digit)`), `./digit/bin/python -m pytest`
passed, but plain `pytest -v` still failed with the same `ModuleNotFoundError`,
using a *different* pytest (9.0.2, with `anyio`/`typeguard`/`hydra-core` plugins
that aren't installed in `digit/`).

### Cause

zsh caches the resolved path of a command the first time it's looked up in a
shell session (command hashing). If `pytest` was run in that terminal tab
before `digit` was activated, zsh cached it to the system binary
(`/Library/Frameworks/Python.framework/Versions/3.13/bin/pytest`). Activating
the venv correctly prepends `digit/bin` to `$PATH`, but zsh doesn't
automatically re-resolve commands it has already cached — so bare `pytest -v`
kept hitting the old cached system binary despite `$PATH` and the prompt both
looking correct.

`./digit/bin/python -m pytest` isn't affected because it's an explicit path,
not a `$PATH` lookup, so there's nothing to cache.

### Fix

Clear the shell's command hash after activating the venv:

```bash
hash -r
pytest -v
```

Alternatively, open a new terminal tab after activating, or just always use
the module form (`python -m pytest`), which resolves against the currently
active interpreter and isn't subject to this caching.
