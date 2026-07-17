"""Generate the versioned precomputed artifact used by the Streamlit demo."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preloaded_artifacts import DEFAULT_ARTIFACT_PATH, load_artifact, save_artifact


def main() -> None:
    path = save_artifact(DEFAULT_ARTIFACT_PATH)
    artifact = load_artifact(path)
    size_mib = path.stat().st_size / (1024 * 1024)
    print(f"Saved {path} ({size_mib:.2f} MiB)")
    print(f"Models: {', '.join(artifact['models'])}")
    print(f"Runtime: {artifact['runtime_versions']}")


if __name__ == "__main__":
    main()
