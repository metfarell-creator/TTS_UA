"""Download core models required for the portable pipeline."""

from __future__ import annotations

from huggingface_hub import snapshot_download

REPOSITORIES = [
    "openai/whisper-large-v3",
    "patriotyk/styletts2-ukrainian",
]


def main() -> None:
    for repo in REPOSITORIES:
        print(f"Downloading {repo} ...")
        snapshot_download(repo_id=repo, local_dir_use_symlinks=False)


if __name__ == "__main__":  # pragma: no cover
    main()
