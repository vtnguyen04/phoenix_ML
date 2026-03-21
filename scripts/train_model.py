"""
DEPRECATED: Moved to examples/credit_risk/train.py

This file is kept as a redirect for backward compatibility.
All new development should use examples/credit_risk/train.py directly.
"""

from examples.credit_risk.train import (  # noqa: F401
    FEATURE_NAMES,
    N_FEATURES,
    train_and_export,
)

if __name__ == "__main__":
    import runpy
    import sys

    print("⚠️  This script has moved to examples/credit_risk/train.py")
    print("    Redirecting...")
    runpy.run_module("examples.credit_risk.train", run_name="__main__")
    sys.exit(0)
