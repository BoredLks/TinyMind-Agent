"""Make `app` importable when running pytest from any working directory."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Tests must never touch the real %APPDATA% database or user skills dir.
os.environ.setdefault("SUPERAGENT_DB_PATH", ":memory:")
os.environ.setdefault(
    "SUPERAGENT_SKILLS_DIR", os.path.join(tempfile.gettempdir(), "superagent_test_skills")
)
