# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-data-pipeline. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-data-pipeline", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Pre-tokenization and special tokens](/course/open/tok-special-tokens?lang=en) and [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_end():
    quick_end = L.choice(
        "quick-end",
        question="Thousands of stories go into one long list of token ids, with the end-of-text id between every two. What does a GPT learn from that id?",
        options={
            "skip": "Nothing: the trainer skips it, as it is not text",
            "right": "That a story can end there, and that the next one starts anew",
            "join": "That the next story goes on where the last one stopped, as one long story",
        },
        correct="right",
        why={
            "skip": "The end-of-text id is a token like any other in the training data: the GPT learns to predict it, and what comes after it. See [Pre-tokenization and special tokens](/course/open/tok-special-tokens?lang=en).",
            "join": "Without the id, a window across two stories would look like one story. With it, the GPT sees where one ends. See [Pre-tokenization and special tokens](/course/open/tok-special-tokens?lang=en).",
            "right": "So the stories of this lesson go into the files as one list of ids, with end-of-text between them.",
        },
    )
    quick_end
    return (quick_end,)


@app.cell(hide_code=True)
def quick_check_len():
    quick_len = L.choice(
        "quick-len",
        question="A Dataset's `__len__` gives 1,000. How many mini-batches does `DataLoader(ds, batch_size=64)` give in one epoch?",
        options={
            "right": "16: 15 of 64 and one of 40",
            "fifteen": "15: the last 40 examples are left out",
            "thousand": "1,000: one for every example",
        },
        correct="right",
        why={
            "fifteen": "A DataLoader keeps the last, smaller mini-batch, unless you ask it not to. See [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en).",
            "thousand": "`__getitem__` gives one example; the DataLoader puts 64 of them in a mini-batch. See [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en).",
            "right": "In this lesson, `__len__` is the number of windows in the files, and `__getitem__` reads one window.",
        },
    )
    quick_len
    return (quick_len,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Until now, every run of [Project: your GPT on your own tokens](/course/open/gpt-train-bpe?lang=en) encoded all the stories again, and held all their ids in memory as a Python list. For 5 MB that works. nanoGPT, a well-known small codebase to train GPT-2, prepares OpenWebText, 8 million web pages: their GPT-2 token ids, about 9 billion, take about 17 GB as 2-byte numbers. At the speed of the course's tokenizer, about 400,000 ids a second, encoding them would take about 6 hours, and they do not fit in the memory of most computers. So a pretraining pipeline encodes the corpus once, writes the ids to files, and during training reads only the windows it needs, straight from the disk.

        **An everyday example.** A library does not copy a book for every reader who wants one page: the books stand on the shelves, numbered, and you take down the one book and open it at the page you need.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           5, here_label="← today", title="Where your project stands: your own LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_batch():
    _m = C.MEASURED
    guess_batch = L.guess(
        "batch",
        question=f"On the course's computer, encoding the {C.number(_m['chars'] / 1e6, 1, L.lang)} million characters of the stories takes about {round(_m['encode'])} seconds. Once the ids are in files, how long do you think it takes to read one mini-batch of {C.BATCH} windows of {C.BLOCK} ids from them?",
        options={
            "same": f"About as long: {round(_m['encode'])} seconds",
            "tenth": "About a tenth of that",
            "right": "A few thousandths of a second",
        },
        reveal=f"**A few thousandths of a second: {C.number(1000 * _m['batch'], 1, L.lang)} ms on the course's computer.** For a mini-batch, only its {C.BATCH} windows of {C.BLOCK + 1} ids are read, however large the files are, and nothing is encoded again. Opening the {_m['shards']} files took {C.number(1000 * _m['open'], 1, L.lang)} ms: nothing is read until a window is asked for.",
    )
    guess_batch
    return (guess_batch,)


@app.cell(hide_code=True)
def text_idea():
    _m = C.MEASURED
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Encode once.** Encode the whole corpus one time, and write the ids to files: from then on, every run, every epoch and every computer reads the ids instead of the text.

        **The smallest type.** An id of a vocabulary of $V$ ids is a number from 0 to $V - 1$. An unsigned 16-bit number, `np.uint16`, holds 0 to 65,535, in 2 bytes: enough for every vocabulary up to $2^{16}$ = 65,536 ids, such as the 1,257 of your tokenizer and the 50,257 of GPT-2. Llama 3, with 128,256, needs `np.uint32`, 4 bytes. NumPy's default type, `int64`, takes 8 bytes per id, and a Python list of ints far more.

        **Token shards.** A large corpus is cut into files of a fixed number of ids each: token shards. Many processes can encode, write and read their own shards at the same time; a broken or lost file costs one shard; and a run can stop after any shard and go on later. A shard holds the ids as bare bytes, one number after the other, without a header: `ids.tofile(path)` writes them, and only the type says how to read them back.

        **Read from the disk.** `np.memmap(path, dtype=np.uint16, mode="r")` opens a shard as an array that stays on the disk: opening costs nothing, and `data[s:s + 129]` reads only those ids. A Dataset can then cut every shard into windows, as `gpt_examples` of [Assemble your GPT](/course/open/gpt-assemble?lang=en) did with a list, and a DataLoader shuffles them.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **token shard** | | one file of a fixed number of token ids, stored as bare numbers of one type; a corpus is many of them | `shard_0.bin` |
        | **memory-mapped file** | | an array whose numbers stay in a file on the disk; indexing it reads only those numbers | `np.memmap(path, dtype=np.uint16, mode="r")` |
        | `uint16` | you-int sixteen | a whole number of 16 bits without a sign: 0 to 65,535, in 2 bytes | `np.uint16` |
        | **to a file** | | write the numbers of an array as bare bytes, one after the other; `np.fromfile` reads them back whole | `arr.tofile(path)` |
        | **a temporary folder** | | a new, empty folder for files that are only needed for a while; it gives back the folder's path | `tempfile.mkdtemp()` |
        """),
        mo.md(r"""
        The picture: what the stories of TinyStories take, as text and as ids, with the course's tokenizer.
        """),
        mo.Html(C.sizes_picture([("the text", _m["chars"] / 1e6), ("the ids, as int64", 8 * _m["ids"] / 1e6), ("the ids, as uint16", 2 * _m["ids"] / 1e6)],
                                {"title": "One corpus, three sizes", "lang": L.lang,
                                 "note": f"{C.number(_m['ids'], 0, L.lang)} ids, in {_m['shards']} shards of up to {C.number(C.SHARD, 0, L.lang)}",
                                 "alt": f"Bars of megabytes: the text {C.number(_m['chars'] / 1e6, 1, L.lang)}; the ids as int64 {C.number(8 * _m['ids'] / 1e6, 1, L.lang)}; the ids as uint16 {C.number(2 * _m['ids'] / 1e6, 1, L.lang)}."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To feed a GPT from token shards:

    1. Encode the corpus once, with end-of-text between the texts.
    2. Choose the smallest type that holds every id: `np.uint16` up to 65,536 ids, else `np.uint32`.
    3. Write the ids, a fixed number at a time, each to a shard of its own.
    4. In training, open every shard with `np.memmap`, read-only.
    5. List the start of every window in every shard, and read a window and the id after it when it is asked for.
    6. Let a DataLoader shuffle the windows and put them in mini-batches.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 2 to 4 with ten made-up ids. `tempfile.mkdtemp()` makes a new, empty folder for temporary files, and gives back its path.
    """)
    return


@app.cell
def example_file():
    import tempfile

    import numpy as np

    _ids = [3, 1256, 7, 42, 0, 999, 12, 5, 64, 300]
    _path = f"{tempfile.mkdtemp()}/example.bin"
    # 2. and 3. as 2-byte numbers, one after the other
    np.array(_ids, dtype=np.uint16).tofile(_path)
    # 4. open the file without reading it, and read three ids
    _data = np.memmap(_path, dtype=np.uint16, mode="r")
    print(len(_data), "ids,", _data.nbytes, "bytes:", _data[4:7])
    # the same bytes read as another type give other numbers
    print(np.fromfile(_path, dtype=np.uint32)[:2])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Ten ids of 2 bytes take 20 bytes on disk, and `_data[4:7]` gives `[  0 999  12]`, the ids at places 4, 5 and 6. Read as `uint32`, the same bytes give other numbers: 3 and 1256 together become $3 + 1256 \cdot 65{,}536 = 82{,}313{,}219$. A shard says nothing about its type, so whoever reads it must know: this lesson keeps it in a small file next to the shards.
    """)
    return


@app.cell(hide_code=True)
def text_torch_rung():
    mo.md(r"""
    ### One rung up: the same in PyTorch

    A GPT needs its windows as tensors of `int64`, which an embedding layer looks up. A slice of a memory map is an array of `uint16`; `.tolist()` makes it a list of Python ints, and `torch.tensor` makes those `int64`, as in [Assemble your GPT](/course/open/gpt-assemble?lang=en). Only that slice is read from the disk.
    """)
    return


@app.cell
def example_tensor():
    import tempfile as _tempfile

    import numpy as _np
    import torch

    _path = f"{_tempfile.mkdtemp()}/example.bin"
    _np.array([3, 1256, 7, 42, 0, 999, 12, 5, 64, 300], dtype=_np.uint16).tofile(_path)
    _data = _np.memmap(_path, dtype=_np.uint16, mode="r")
    # a window of 4 ids and the one after it, as a tensor
    _ids = torch.tensor(_data[2:7].tolist())
    print(_ids[:-1], _ids[1:], _ids.dtype)
    return


@app.cell(hide_code=True)
def text_after_torch():
    mo.md(r"""
    | NumPy | PyTorch |
    |---|---|
    | `data[s:s + block + 1]`, an array of `uint16` | `torch.tensor(data[s:s + block + 1].tolist())`, of `int64` |
    | the window and its target: `w[:-1]`, `w[1:]` | the same: `ids[:-1]`, `ids[1:]` |

    The window `[7, 42, 0, 999]` has as target `[42, 0, 999, 12]`: every id the one after it, as in [Assemble your GPT](/course/open/gpt-assemble?lang=en).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_write():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Write the shards

        Complete `write_shards(ids, vocab, folder, size)`: steps 2 and 3 of the recipe. Choose `np.uint16` when `vocab`, the number of different ids, is at most `2 ** 16`, and else `np.uint32`. Then write every `size` ids to a file of its own in `folder`, named `shard_0.bin`, `shard_1.bin`, and so on; the last shard holds what is left. It gives back the list of the files' paths. The check writes a few shards in a temporary folder, and reads them back.
        """),
        L.task("Make `write_shards` write every `size` ids, in the smallest type, to a numbered shard."),
    ])
    return


@app.cell(hide_code=True)
def exercise_write():
    box_write = L.code_box("write-shards", lines=16)
    box_write
    return (box_write,)


@app.cell(hide_code=True)
def check_write(box_write):
    result_write = L.check("write-shards", box_write.value)
    result_write
    return (result_write,)


@app.cell(hide_code=True)
def hints_write():
    L.hints(
        "write-shards",
        [
            "`range(0, len(ids), size)` gives the place where every shard starts; `ids[start:start + size]` its ids, also for the last, shorter one.",
            "The name of the next shard is its number, `len(paths)`, in an f-string. The worked example shows how one shard is written.",
        ],
        why="Two bytes per id and no header make a shard as small as it can be, and every id sits at a place you can compute: id i of a shard starts at byte 2i. That is what lets `np.memmap` read one window without the rest.",
    )
    return


@app.cell(hide_code=True)
def text_demo_write():
    mo.vstack([
        mo.md(f"""
        ### Your stories as shards

        Once exercise 1 is right, the course encodes all {C.number(C.MEASURED['chars'], 0, L.lang)} characters of TinyStories once, with your `encode_fast` of [Project: your GPT on your own tokens](/course/open/gpt-train-bpe?lang=en) and the merge rules of your GPT on tokens (or of the course's, when you have not trained one), and your `write_shards` writes them in shards of {C.number(C.SHARD, 0, L.lang)} ids to `work/projects/llm/{C.FOLDER}/`. Later lessons train from there.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def demo_write(box_write, result_write):
    if result_write.state != "correct":
        written = None
        _out = mo.md("*Finish exercise 1 first: the demo runs your `write_shards`.*")
    else:
        _r = C.write_demo(run_code(box_write.value).names["write_shards"])
        written = _r if isinstance(_r, dict) else None
        if _r == "run":
            _out = mo.md("*Your `encode_fast` in `work/projects/llm/fast_encode.py` stops with an error. Check its exercise again, or delete the file to use the course's version.*")
        else:
            _whose = "your GPT" if _r["source"] == "yours" else "the course's GPT"
            _out = mo.md(f"""With the merge rules of {_whose}, the {C.number(_r['chars'] / 1e6, 1, L.lang)} million characters become {C.number(_r['ids'], 0, L.lang)} ids, in {C.number(_r['seconds'], 1, L.lang)} seconds. Your `write_shards` wrote them into {_r['shards']} shards, {C.number(_r['bytes'] / 1e6, 1, L.lang)} MB together: {C.number(_r['bytes'] / _r['ids'], 0, L.lang)} bytes per id, against {C.number(_r['text_bytes'] / _r['ids'], 1, L.lang)} bytes of text per id. The largest id is {_r['top']}, the end-of-text id.""")
    _out
    return (written,)


@app.cell(hide_code=True)
def text_exercise_windows():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Windows from the disk

        Complete the class `TokenWindows(paths, block, dtype)`: steps 4 and 5. Its `__init__` opens every file of `paths` with `np.memmap`, read-only, and lists the place of every window as `(k, s)`: shard `k`, start `s`. The windows of a shard lie side by side, `block` ids apart, from 0, as long as one more id follows the window: a window never runs into the next shard. `__len__` gives the number of windows, and `__getitem__(i)` reads window `i` and the id after it, as a tensor of `int64`, and gives back the window and its target. The check writes two small shards and asks your class for its windows.
        """),
        L.task("Make `TokenWindows` read every window of every shard, and its target, from the disk."),
    ])
    return


@app.cell(hide_code=True)
def exercise_windows():
    box_windows = L.code_box("token-windows", lines=22)
    box_windows
    return (box_windows,)


@app.cell(hide_code=True)
def check_windows(box_windows):
    result_windows = L.check("token-windows", box_windows.value)
    result_windows
    return (result_windows,)


@app.cell(hide_code=True)
def hints_windows():
    L.hints(
        "token-windows",
        [
            "A window starting at `s` needs ids `s` to `s + block`, so `s + block` must be a place in the shard: `s < len(shard) - block`. `range` with a step of `block` gives such starts.",
            "In `__getitem__`, slice the shard `self.shards[k]` from `s`, `self.block + 1` ids long, as in the rung above.",
        ],
        why="Listing only `(k, s)` costs a few bytes per window; the ids themselves stay on the disk until a DataLoader asks for them. That is how a corpus of many gigabytes trains on a computer with less memory.",
    )
    return


@app.cell(hide_code=True)
def demo_read(box_windows, result_windows, written):
    mo.stop(result_windows.state != "correct", mo.md("*Finish exercise 2 first: the demo runs your `TokenWindows`.*"))
    _r = None if written is None else C.read_demo(run_code(box_windows.value).names["TokenWindows"])
    if _r is None:
        _out = mo.md("*This demo reads the shards that the demo of exercise 1 writes: it runs once that demo has written them.*")
    else:
        _same = ("They are exactly the windows that `gpt_examples` makes of every shard." if _r["same"] else
                 "They are not the windows that `gpt_examples` makes of every shard: check exercise 2 again.")
        _text = "" if _r["text"] is None else f"\n\nThe first window of that mini-batch, decoded:\n\n> {_r['text'][:400].replace(chr(10), ' ')}"
        _out = mo.md(f"""Your `TokenWindows` opens the shards in {C.number(1000 * _r['opened'], 1, L.lang)} ms and finds {C.number(_r['windows'], 0, L.lang)} windows of {C.BLOCK} ids. A DataLoader that shuffles them reads one mini-batch of shape {_r['shape']} in {C.number(1000 * _r['batch'], 1, L.lang)} ms. {_same}{_text}""")
    _out
    return


@app.cell(hide_code=True)
def text_exercise_shards():
    mo.md(r"""
    ### Exercise 3. Why shards?
    """)
    return


@app.cell(hide_code=True)
def choice_shards_cell():
    choice_shards = L.choice(
        "why-shards",
        question="A corpus of 2 trillion token ids is stored as thousands of shards, not as one file. Why?",
        options={
            "memmap": "Because `np.memmap` cannot open a file of more than a few gigabytes, so a corpus must be split",
            "right": "So that many processes can each work on their own shards, and a broken file costs one shard",
            "small": "Because a shard of fewer ids needs fewer bytes per id, so the corpus as a whole gets smaller",
        },
        correct="right",
        why={
            "memmap": "On a 64-bit computer, `np.memmap` opens files far larger than the computer's memory: that is what it is for. The reason is the work around the files.",
            "small": "The bytes per id depend only on the type, 2 for `uint16`, however many ids a shard holds.",
            "right": "Encoding, writing and reading all go in parallel, shard by shard, and a run can stop and go on at any shard.",
        },
    )
    choice_shards
    return (choice_shards,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The stories of [Project: your GPT on your own tokens](/course/open/gpt-train-bpe?lang=en), now encoded once, with the end-of-text of [Pre-tokenization and special tokens](/course/open/tok-special-tokens?lang=en) between them; and the Dataset of [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en), now reading from the disk. A real pipeline cleans its corpus before this step, as [Keeping test data out of training data](/course/open/llm-decontamination?lang=en) does with test questions. Later lessons train from these shards.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_write, result_windows, choice_shards):
    L.wrap_up(result_write, result_windows, choice_shards)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
