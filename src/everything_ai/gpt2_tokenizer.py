# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""GPT-2's tokenizer without internet. tiktoken's "gpt2" encoding downloads two files the first time and keeps them
in a cache, under the SHA-1 of their address. The course ships the same two files in data/gpt2-tokenizer/
(data/README.md says where they come from), and puts them in that cache before tiktoken looks, so that
`tiktoken.get_encoding("gpt2")` in a lesson works offline and gives the same tokens everywhere."""
import filecmp
import hashlib
import os
import shutil
from pathlib import Path

from everything_ai import paths

FILES = paths.ROOT / "data" / "gpt2-tokenizer"
_ADDRESSES = {   # where tiktoken 0.14 looks for each file (tiktoken_ext/openai_public.py), and the course's copy
    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe": "vocab.bpe",
    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json": "encoder.json",
}


def offline() -> None:
    """Put the course's copies of GPT-2's tokenizer files in tiktoken's cache, unless the same bytes are there
    already. tiktoken checks their SHA-256 itself."""
    if not os.environ.get("TIKTOKEN_CACHE_DIR"):   # not set, or "", which turns tiktoken's cache off
        os.environ["TIKTOKEN_CACHE_DIR"] = str(paths.WORK / "cache" / "tiktoken")
    cache = Path(os.environ["TIKTOKEN_CACHE_DIR"])
    try:
        cache.mkdir(parents=True, exist_ok=True)
        for address, name in _ADDRESSES.items():
            source, target = FILES / name, cache / hashlib.sha1(address.encode()).hexdigest()
            if target.is_file() and filecmp.cmp(source, target, shallow=False):
                continue
            part = target.with_name(f"{target.name}.{os.getpid()}.part")
            shutil.copyfile(source, part)
            os.replace(part, target)   # whole or not at all, also when two lessons open at the same time
    except OSError:
        pass   # a cache folder the course cannot write to: tiktoken then downloads the files, as without the course
