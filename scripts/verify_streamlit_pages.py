"""Smoke-test every dashboard route with Streamlit's application harness."""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest


PRELOADED_PAGES = (
    "Overview",
    "EDA",
    "Model Comparison",
    "Decision Boundaries",
    "Confusion Matrices",
    "ROC / AUC",
    "Precision-Recall",
    "Learning Curves",
    "Live Prediction",
)


def _radio(app: AppTest, label: str):
    return next(item for item in app.radio if item.label == label)


def _assert_clean(app: AppTest, route: str) -> None:
    if app.exception:
        messages = "; ".join(str(item.value) for item in app.exception)
        raise RuntimeError(f"{route} raised a Streamlit exception: {messages}")


def main() -> None:
    app = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=60)
    started = time.perf_counter()
    app.run()
    _assert_clean(app, "initial load")
    print(f"initial load: {time.perf_counter() - started:.3f}s")

    for page in PRELOADED_PAGES:
        started = time.perf_counter()
        _radio(app, "Navigate").set_value(page)
        app.run()
        _assert_clean(app, page)
        print(f"{page}: {time.perf_counter() - started:.3f}s")

    for mode in ("Custom Dataset", "AI Dataset Assistant"):
        started = time.perf_counter()
        _radio(app, "Mode").set_value(mode)
        app.run()
        _assert_clean(app, mode)
        print(f"{mode} landing page: {time.perf_counter() - started:.3f}s")


if __name__ == "__main__":
    main()
