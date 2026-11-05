"""Test files that need the training extra are not collected on an app-only install."""

import importlib.util

collect_ignore = [
    file
    for file, module in (("test_teacher.py", "torch"), ("test_hpo.py", "optuna"))
    if importlib.util.find_spec(module) is None
]
