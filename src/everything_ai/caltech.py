# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Caltech-101, the photo set of the vision branch: photos of objects in 101 classes, collected from the web in
2003 by Fei-Fei Li, Marco Andreetto, Marc'Aurelio Ranzato and Pietro Perona, CC BY 4.0 on CaltechDATA
(doi:10.22002/D1.20086). The course does not ship it: everything_ai.hub downloads its zip file once, at its source,
and `unpack` puts the photos in work/cache/caltech-101/, one folder per class. The two classes of faces are left out
(the course shows no people), and so is BACKGROUND_Google, which is no class but random photos from the web. Other
classes can still show a person here and there, so a lesson uses only photos that were looked at one by one
(CHECKED), and the few among them with a person, or part of one, are left out as well (PEOPLE). The course trains
no model on it that it shares: the rights to the photos lie with their photographers (course/route/bronnen.yaml,
caltech-101-download-only).

Next to the photos, the zip file holds an outline of the object on every photo, drawn by hand by the makers of
Caltech-101. `outlines` gives those of the photos in CHECKED, from a file `unpack_outlines` makes once."""
import gzip
import json
import os
import re
import shutil
import struct
import tarfile
import tempfile
import time
import zipfile
import zlib
from pathlib import Path

from everything_ai import hub, paths

ZIP = ("caltechdata/mzrjq-6wc02", "caltech-101.zip")
MEMBER = "caltech-101/101_ObjectCategories.tar.gz"   # the photos, inside the zip file
OUTLINES = "caltech-101/Annotations.tar"   # and the outlines, one MATLAB file per photo
LEFT_OUT = {"Faces", "Faces_easy", "BACKGROUND_Google"}
# The first 35 photos of each of these classes were looked at one by one, for people: the photos the lessons use.
CHECKED = ("chandelier", "dragonfly", "ewer", "lotus", "nautilus", "rooster", "starfish", "sunflower", "water_lilly",
           "windsor_chair")
PEOPLE = {"sunflower/image_0004.jpg", "sunflower/image_0022.jpg", "ewer/image_0023.jpg"}   # what that found: never unpacked
CLASSES = 99      # the classes the course keeps: 101 and BACKGROUND_Google, without these three
PHOTOS = 7804     # the photos in them, without PEOPLE
_NAME = re.compile(r"101_ObjectCategories/([A-Za-z_]+)/(image_\d{4}\.jpg)")
_OUTLINE = re.compile(r"Annotations/([A-Za-z_]+)/annotation_(\d{4})\.mat")
_LARGEST = 1_000_000   # bytes; the largest photo of the classes the course keeps has 45 kB


def folder() -> Path:
    return paths.WORK / "cache" / "caltech-101"


def unpacked() -> bool:
    """Whether the photos are unpacked, as this version of the course unpacks them: a folder appears only once all
    its photos are in it, and "done" says how many there are."""
    try:
        return (folder() / "done").read_text().strip() == str(PHOTOS)
    except OSError:
        return False


def unpack(progress=None) -> Path:
    """Unpack the downloaded zip file into work/cache/caltech-101/, once, and give that folder. Only photos with the
    name of a photo in a class folder are taken, and none of the classes in LEFT_OUT or the photos in PEOPLE; the
    archive is read as a stream, so nothing in it is ever run or written outside that folder. `progress(done)` hears
    about every 500 photos. It raises FileNotFoundError when the zip file is not downloaded yet: hub.download(*ZIP)
    gets it. A zip file that turns out to be damaged is deleted, so that the next try downloads it again, and gives an
    OSError."""
    if unpacked():
        return folder()
    source = hub.pinned(*ZIP)
    if source is None:
        raise FileNotFoundError("Caltech-101 is not downloaded yet: everything_ai.hub.download(*everything_ai.caltech.ZIP)")
    folder().parent.mkdir(parents=True, exist_ok=True)
    for old in folder().parent.glob("caltech-101.*.part"):   # left by a lesson that was stopped while it unpacked
        if time.time() - old.stat().st_mtime > 3600:
            shutil.rmtree(old, ignore_errors=True)
    tmp = Path(tempfile.mkdtemp(dir=folder().parent, prefix="caltech-101.", suffix=".part"))
    try:
        done = 0
        try:
            with zipfile.ZipFile(source) as z, z.open(MEMBER) as packed, gzip.open(packed) as raw, \
                    tarfile.open(fileobj=raw, mode="r|") as photos:
                for member in photos:
                    found = _NAME.fullmatch(member.name)
                    if (not member.isfile() or not found or found[1] in LEFT_OUT
                            or f"{found[1]}/{found[2]}" in PEOPLE or member.size > _LARGEST):
                        continue
                    (tmp / found[1]).mkdir(exist_ok=True)
                    with photos.extractfile(member) as f:
                        (tmp / found[1] / found[2]).write_bytes(f.read(_LARGEST + 1))
                    done += 1
                    if progress is not None and done % 500 == 0:
                        progress(done)
        except (zipfile.BadZipFile, tarfile.TarError, EOFError, zlib.error, KeyError) as e:
            source.unlink(missing_ok=True)
            raise OSError(f"the zip file of Caltech-101 is damaged, so the course deleted it: {e}") from e
        if done != PHOTOS:
            raise OSError(f"Caltech-101 holds {done} photos, not {PHOTOS}")
        (tmp / "done").write_text(f"{done}\n")
        for p in [tmp, *tmp.rglob("*")]:   # mkdtemp makes a folder only its owner may read
            os.chmod(p, 0o755 if p.is_dir() else 0o644)
        if unpacked():   # another lesson unpacked the photos meanwhile: keep its folder, which may be in use
            shutil.rmtree(tmp, ignore_errors=True)
            return folder()
        shutil.rmtree(folder(), ignore_errors=True)   # what is left of an earlier try that broke off
        os.replace(tmp, folder())   # in one step: a lesson never sees half the photos
        return folder()
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def classes() -> list[str]:
    """The names of the classes, in alphabetical order."""
    return sorted(p.name for p in folder().iterdir() if p.is_dir())


def photos(name: str) -> list[Path]:
    """The photos of a class, in the order of their numbers."""
    return sorted((folder() / name).glob("image_*.jpg"))


def checked(name: str, n: int = 30) -> list[Path]:
    """The first n photos of a class in CHECKED: photos without people, looked at one by one (at most 35 minus
    what PEOPLE leaves out)."""
    if name not in CHECKED or n > 32:
        raise ValueError(f"only the first photos of the classes in CHECKED were looked at: {name}, {n}")
    return photos(name)[:n]


def _outlines_file() -> Path:
    return paths.WORK / "cache" / "caltech-101-outlines.json"


# the types of numbers in a MATLAB file of version 5, and how struct reads them
_NUMBERS = {1: "b", 2: "B", 3: "h", 4: "H", 5: "i", 6: "I", 7: "f", 9: "d", 12: "q", 13: "Q"}
_MATRIX, _COMPRESSED = 14, 15


def _elements(data: bytes):
    """The elements of a MATLAB file of version 5, after its header, or of a matrix in it, as (type, bytes); a
    compressed element is unpacked. Little-endian only, as all files of Caltech-101 are."""
    at = 0
    while at + 8 <= len(data):
        kind, size = struct.unpack_from("<II", data, at)
        if kind >> 16:   # a small element: type and size in one number, the bytes in the next four
            kind, size = kind & 0xFFFF, kind >> 16
            yield kind, data[at + 4:at + 4 + size]
            at += 8
            continue
        body = data[at + 8:at + 8 + size]
        if len(body) != size:
            raise ValueError("a MATLAB file that ends too soon")
        if kind == _COMPRESSED:
            yield from _elements(zlib.decompressobj().decompress(body, 1_000_000))
            at += 8 + size   # a compressed element is not padded
        else:
            yield kind, body
            at += 8 + size + (-size % 8)   # every other element is padded to a multiple of 8 bytes


def _read_mat(data: bytes) -> dict:
    """The matrices of numbers in a MATLAB file of version 5, by name, each as a list of its columns. The course's
    own reader, for the few files of Caltech-101 it needs: nothing in the file is ever run."""
    if len(data) < 128 or data[126:128] != b"IM":
        raise ValueError("not a little-endian MATLAB file of version 5")
    found = {}
    for kind, body in _elements(data[128:]):
        if kind != _MATRIX:
            continue
        parts = list(_elements(body))
        if len(parts) < 4 or parts[3][0] not in _NUMBERS:
            continue   # a matrix of something else than numbers
        rows, cols = struct.unpack_from("<ii", parts[1][1])
        name = parts[2][1].decode("ascii")
        code = _NUMBERS[parts[3][0]]
        values = struct.unpack(f"<{len(parts[3][1]) // struct.calcsize(code)}{code}", parts[3][1])
        if len(values) != rows * cols:
            raise ValueError(f"matrix {name} has {len(values)} numbers, not {rows} by {cols}")
        found[name] = [list(values[c * rows:(c + 1) * rows]) for c in range(cols)]
    return found


def _outline(data: bytes) -> list[list[float]]:
    """The outline of one annotation file, as points [x, y] in pixels of the photo, from its top-left pixel at
    [0, 0]. The file keeps the box around the object, box_coord, as its top, bottom, left and right, counted from 1,
    and the outline, obj_contour, relative to the top-left corner of that box."""
    mat = _read_mat(data)
    (top,), (_,), (left,), (_,) = mat["box_coord"]
    return [[round(x + left - 1, 2), round(y + top - 1, 2)] for x, y in mat["obj_contour"]]


def unpack_outlines() -> Path:
    """Read the outlines of the photos in CHECKED that were looked at, the first 35 of each class, from the downloaded
    zip file into one file, once, and give that file. The outlines of the photos in PEOPLE are left out, as those
    photos are. It raises FileNotFoundError when the zip file is not downloaded yet. A zip file that turns out to be
    damaged is deleted, so that the next try downloads it again, and gives an OSError."""
    if _outlines_file().exists():
        return _outlines_file()
    source = hub.pinned(*ZIP)
    if source is None:
        raise FileNotFoundError("Caltech-101 is not downloaded yet: everything_ai.hub.download(*everything_ai.caltech.ZIP)")
    found = {}
    try:
        with zipfile.ZipFile(source) as z, z.open(OUTLINES) as raw, tarfile.open(fileobj=raw, mode="r|") as files:
            for member in files:
                name = _OUTLINE.fullmatch(member.name)
                if (not member.isfile() or not name or name[1] not in CHECKED or int(name[2]) > 35
                        or member.size > _LARGEST):
                    continue
                photo = f"{name[1]}/image_{name[2]}.jpg"
                if photo not in PEOPLE:
                    with files.extractfile(member) as f:
                        found[photo] = _outline(f.read(_LARGEST + 1))
    except (zipfile.BadZipFile, tarfile.TarError, EOFError, zlib.error, KeyError, ValueError, struct.error) as e:
        source.unlink(missing_ok=True)   # damaged: the next try downloads it again, as unpack does
        raise OSError(f"the outlines of Caltech-101 could not be read: {e}") from e
    _outlines_file().parent.mkdir(parents=True, exist_ok=True)
    tmp = _outlines_file().with_suffix(f".{os.getpid()}.part")
    tmp.write_text(json.dumps(found, separators=(",", ":")))
    os.replace(tmp, _outlines_file())   # in one step: a lesson never reads half the file
    return _outlines_file()


def outlines_ready() -> bool:
    """The outlines are unpacked, or the zip file they come from is there to unpack them from."""
    return _outlines_file().exists() or hub.pinned(*ZIP) is not None


def outlines(name: str) -> dict[str, list[list[float]]]:
    """The outlines of the photos of a class in CHECKED, by the photo's file name, such as "image_0001.jpg": each a
    list of points [x, y] in pixels, around the object."""
    if name not in CHECKED:
        raise ValueError(f"the course reads only the outlines of the classes in CHECKED: {name}")
    every = json.loads(unpack_outlines().read_text())
    return {k.split("/")[1]: v for k, v in every.items() if k.split("/")[0] == name}
