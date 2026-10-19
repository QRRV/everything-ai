# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The course's own models on Hugging Face, the model hub the course uses. A file is downloaded once from its fixed
address, without an account, into work/cache/hub/, and only kept when its SHA-256 is that of the course's own copy
in data/ (data/README.md says how that copy was made). Without a connection, or when the hub's file is not that
file, the course's copy is used: a lesson works offline and gives the same numbers everywhere. After one failed try,
the same Python process does not wait for that repository again. Set EAI_OFFLINE to never try the hub, as the
tests do.

Other models, too large for the course to ship, are downloaded from a repository on the hub (PINNED): once,
when a learner asks for it, from a fixed commit, and kept only when the SHA-256 is the one the course pinned. After
that the lesson works offline. Of a large file, a lesson can pin only its first bytes (`first`), such as the
beginning of a compressed shard of web pages: a learner then downloads only those."""
import hashlib
import http.client
import os
import tempfile
import urllib.request
from pathlib import Path

from everything_ai import paths

HUB = "https://huggingface.co"
TIMEOUT = 10   # seconds to wait for the hub before the course's copy is used

# Per repository on the hub: each file and the course's own copy of it.
REPOS = {
    "QRRV/everything-ai-qmnist-cnn": {
        "model.pt": paths.ROOT / "data" / "course-qmnist-cnn.pt",
        "README.md": paths.ROOT / "data" / "course-qmnist-cnn.md",
    },
}


# Per file the course downloads from its own repository: the commit it comes from, its SHA-256 and its size in bytes.
PINNED = {
    ("openai-community/gpt2", "model.safetensors"): {
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "sha256": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
        "size": 548_105_171,
    },
    # the first 20 MB of a shard of C4 before its cleaning: raw web pages for llm-quality-filter (course/route/bronnen.yaml)
    ("datasets/allenai/c4", "en.noclean/c4-train.00000-of-07168.json.gz"): {
        "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        "sha256": "2b74106294af67067b72ecd4a4075e6e386382a546bbf7f93960b5f90b7e747a",
        "size": 20_000_000,
        "first": True,
    },
    # the first 200 MB of the training stories of TinyStories: the data of llm-pretrain-capstone (course/route/bronnen.yaml)
    ("datasets/roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt"): {
        "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "sha256": "d4ab5db7423aee03fecad9dc029119c540259518dd73858bb58ef640c05cf160",
        "size": 200_000_000,
        "first": True,
    },
}
CHUNK = 1 << 20   # bytes per read while downloading


def address(repo: str, filename: str) -> str:
    """The fixed address of a file on the hub: the newest version on the branch main."""
    return f"{HUB}/{repo}/resolve/main/{filename}"


def page(repo: str) -> str:
    """The web page of a repository on the hub, with its model card."""
    return f"{HUB}/{repo}"


_UNREACHABLE: set[str] = set()   # repositories this process could not download from: it does not wait for them again


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(repo: str, filename: str) -> tuple[Path, str]:
    """A file of one of the course's repositories, and where it came from: "hub" when it was downloaded from the hub,
    now or earlier, and "course" when the course's own copy is used."""
    copy = REPOS[repo][filename]
    want = _sha256(copy)
    cached = paths.WORK / "cache" / "hub" / repo / filename
    if cached.is_file() and _sha256(cached) == want:
        return cached, "hub"
    if os.environ.get("EAI_OFFLINE") or repo in _UNREACHABLE:
        return copy, "course"
    try:
        with urllib.request.urlopen(address(repo, filename), timeout=TIMEOUT) as response:
            data = response.read(copy.stat().st_size + 1)   # never more than the course's file and one byte
    except Exception:   # no connection, no such repository yet, or the hub does not answer
        _UNREACHABLE.add(repo)
        return copy, "course"
    if hashlib.sha256(data).hexdigest() != want:
        return copy, "course"
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cached.parent, prefix=filename + ".", suffix=".part")   # one of its own per lesson
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o644)   # mkstemp makes a file only its owner may read
        os.replace(tmp, cached)   # in one step, also over an old file: another lesson never reads half a file
    except OSError:   # a work folder the course cannot write to: the file is still right, only not kept
        return copy, "hub"
    return cached, "hub"


class NotPinned(OSError):
    """The hub gave another file than the one the course pinned: another size or SHA-256."""


def pinned_address(repo: str, filename: str) -> str:
    """The address of a pinned file on the hub: at its fixed commit, so that it never changes."""
    return f"{HUB}/{repo}/resolve/{PINNED[(repo, filename)]['revision']}/{filename}"


def _pinned_path(repo: str, filename: str) -> Path:
    pin = PINNED[(repo, filename)]
    name = f"{filename}.first{pin['size']}" if pin.get("first") else filename   # only the beginning of the file
    return paths.WORK / "cache" / "hub" / repo / pin["revision"] / name


def pinned(repo: str, filename: str) -> Path | None:
    """The downloaded file, when it is there with its full size, else None. It looks only on disk: the SHA-256 was
    checked when the file was downloaded."""
    path = _pinned_path(repo, filename)
    return path if path.is_file() and path.stat().st_size == PINNED[(repo, filename)]["size"] else None


def download(repo: str, filename: str, progress=None) -> Path:
    """Download a pinned file once, into work/cache/hub/, and give its path. `progress(done, total)` hears about every
    megabyte. A file of another size or SHA-256 is thrown away: then it raises NotPinned, and without a connection, or
    when it breaks off, another OSError. EAI_OFFLINE makes it never try."""
    have = pinned(repo, filename)
    if have is not None:
        return have
    if os.environ.get("EAI_OFFLINE"):
        raise OSError("offline: EAI_OFFLINE is set")
    pin = PINNED[(repo, filename)]
    target = _pinned_path(repo, filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".part")
    try:
        digest, done = hashlib.sha256(), 0
        first = pin.get("first", False)
        request = pinned_address(repo, filename)
        if first:   # ask for those bytes only
            request = urllib.request.Request(request, headers={"Range": f"bytes=0-{pin['size'] - 1}"})
        # of a file pinned by its first bytes, exactly those: also when the hub sends the whole file anyway
        wanted = pin["size"] if first else pin["size"] + 1
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(request, timeout=60) as response:
            while done < wanted:   # never more than the pinned size and one byte
                chunk = response.read(min(CHUNK, wanted - done))
                if not chunk:
                    break
                f.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, pin["size"])
        if done != pin["size"] or digest.hexdigest() != pin["sha256"]:
            raise NotPinned(f"the file from the hub is not the pinned one ({done} bytes)")
        os.chmod(tmp, 0o644)   # mkstemp makes a file only its owner may read; a download is for whoever uses the work folder
        os.replace(tmp, target)   # in one step: another lesson never reads half a file
        return target
    except BaseException as e:
        Path(tmp).unlink(missing_ok=True)
        if isinstance(e, http.client.HTTPException):   # a response that broke off in the middle
            raise ConnectionError(f"the connection broke off ({type(e).__name__})") from e
        raise
