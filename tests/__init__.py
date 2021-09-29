import os

_TEST_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)

if _PROJECT_ROOT not in os.getenv("PYTHONPATH", ""):
    splitter = ":" if os.environ.get("PYTHONPATH", "") else ""
    os.environ[
        "PYTHONPATH"
    ] = f'{_PROJECT_ROOT}{splitter}{os.environ.get("PYTHONPATH", "")}'
