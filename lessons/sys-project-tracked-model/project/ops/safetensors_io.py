# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT-0
# A safetensors file, written and read by hand: the Everything AI course put the format of an
# earlier lesson here. It writes the same bytes as the library safetensors: the names sorted,
# the JSON without spaces.
# Een safetensors-bestand, met de hand geschreven en gelezen: de Everything AI-course zette hier
# het formaat van een eerdere les neer. Het schrijft dezelfde bytes als de library safetensors:
# de namen gesorteerd, de JSON zonder spaties.
import json
import pathlib

import torch


def save_file(state, path):
    # 8 bytes with the length of the header, the header, then the numbers, 4 bytes each
    # 8 bytes met de lengte van de header, de header, dan de getallen, 4 bytes per getal
    names = sorted(state)
    header = {}
    start = 0
    for name in names:
        end = start + state[name].numel() * 4
        shape = list(state[name].shape)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [start, end]}
        start = end
    text = json.dumps(header, separators=(",", ":")).encode()
    while len(text) % 8:
        text += b" "
    numbers = b"".join(state[name].detach().float().contiguous().numpy().astype("<f4").tobytes()
                       for name in names)
    with open(path, "wb") as f:
        f.write(len(text).to_bytes(8, "little") + text + numbers)


def load_file(path):
    # every tensor of the file, by name; reading takes slices of the bytes and runs nothing
    # elke tensor van het bestand, per naam; lezen neemt slices van de bytes en voert niets uit
    data = pathlib.Path(path).read_bytes()
    n = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + n])
    state = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        numbers = bytearray(data[8 + n + start:8 + n + end])
        state[name] = torch.frombuffer(numbers, dtype=torch.float32).reshape(info["shape"])
    return state
