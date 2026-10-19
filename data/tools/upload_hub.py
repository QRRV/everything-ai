# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Puts the course's own models on Hugging Face, in the repositories of everything_ai.hub.REPOS, each file from the
course's copy in data/. Needs a token with write access to those repositories in the environment variable HF_TOKEN,
and the package huggingface_hub, which the course itself does not need. Run from the course folder:

    uv run --with huggingface_hub python data/tools/upload_hub.py

Afterwards it downloads every file again, without the token, and checks that its SHA-256 is that of the copy."""
import hashlib
import os
import urllib.request

from everything_ai import hub


def main():
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a token with write access to: " + ", ".join(hub.REPOS))
    api = HfApi(token=token)
    for repo, files in hub.REPOS.items():
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        for filename, copy in files.items():
            api.upload_file(path_or_fileobj=str(copy), path_in_repo=filename, repo_id=repo, repo_type="model",
                            commit_message=f"{filename} from the course's data/{copy.name}")
        for filename, copy in files.items():
            with urllib.request.urlopen(hub.address(repo, filename), timeout=60) as response:
                same = hashlib.sha256(response.read()).hexdigest() == hashlib.sha256(copy.read_bytes()).hexdigest()
            print(repo, filename, "the same as the course's copy" if same else "NOT the same as the course's copy")


if __name__ == "__main__":
    main()
