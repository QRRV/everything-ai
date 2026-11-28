# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-containers. Route: course/route/lessen/systemen.yaml. Checks, the two Dockerfiles, their seconds and megabytes, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-containers", "en")
    C = L.checks   # the two Dockerfiles, the layers of their images, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The same packages everywhere: environments and pinning](/course/open/sys-environments?lang=en) and [Someone else's computer: the cloud](/course/open/sys-cloud?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_lock():
    quick_lock = L.choice(
        "quick-lock",
        question="What does a lock file such as `uv.lock` record?",
        options={
            "named": "Only the versions of the packages you name in pyproject.toml",
            "right": "The exact version of every package, and of the packages they need",
            "system": "Every package, the exact version of Python, and the system libraries of your computer",
            "newest": "The newest version that each range allows on the day you install it",
        },
        correct="right",
        why={
            "right": "A lock file records the exact version of every package, with the packages they need, and `uv sync` installs the same from it everywhere. It leaves out Python itself, the system libraries some packages use, and the files and the start command of your service: a container packs those too.",
            "named": "Pinning the packages you name is not enough, since they need packages of their own. What does a lock file add? See [The same packages everywhere: environments and pinning](/course/open/sys-environments?lang=en).",
            "system": "A lock file is about Python packages: Python itself and the libraries of the operating system stay outside it. What does it record about the packages? See [The same packages everywhere: environments and pinning](/course/open/sys-environments?lang=en).",
            "newest": "A range does that, and so it gives other versions over time. What does a lock file record instead? See [The same packages everywhere: environments and pinning](/course/open/sys-environments?lang=en).",
        },
    )
    quick_lock
    return (quick_lock,)


@app.cell(hide_code=True)
def quick_check_iaas():
    quick_iaas = L.choice(
        "quick-iaas",
        question="You rent a virtual machine with a GPU, as IaaS. Who installs Python, the packages and your service on it?",
        options={
            "right": "You do: the provider gives a bare machine with its network",
            "provider": "The provider, since you pay for the machine by the hour",
            "ready": "Nobody: with IaaS, everything is installed and ready to use, as with SaaS",
            "google": "Google, since a virtual machine works the way Colab does",
        },
        correct="right",
        why={
            "right": "With IaaS, you rent bare computers and install the rest yourself. Doing that by hand on every rented machine is slow, and a step forgotten on one of them breaks the service there. A container packs it all once.",
            "provider": "The provider rents out the machine; with IaaS, the software on it is up to you. Who looks after what in IaaS? See [Someone else's computer: the cloud](/course/open/sys-cloud?lang=en).",
            "ready": "Finished software is SaaS. What does IaaS rent out, and what is left for you? See [Someone else's computer: the cloud](/course/open/sys-cloud?lang=en).",
            "google": "Colab is a service of its own, with notebooks ready to use. What does a bare virtual machine come with? See [Someone else's computer: the cloud](/course/open/sys-cloud?lang=en).",
        },
    )
    quick_iaas
    return (quick_iaas,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A service like that of the project *ops* runs fine on your own computer. To run it on a rented server, the server needs the same Python, the same packages, the model, the code, and the command that starts it all. A lock file fixes the packages, but not Python itself, not the system libraries some packages need, and not the files of the service. Installing all of it by hand on every server takes time, and one step forgotten on one server makes the service fail there and not on your computer: the well-known *it works on my machine*. A container packs everything in one box that runs the same everywhere.

    **An everyday example.** Ships used to be loaded crate by crate and sack by sack. A shipping container is a steel box of a standard size: the ship, the crane and the truck handle every box the same way, whatever is inside. A software container does the same for programs: a server only has to run containers, whatever is in them.
    """)
    return


@app.cell(hide_code=True)
def predict_build():
    guess_build = L.guess(
        "build",
        question="A Dockerfile first copies the whole project into the image, then installs the packages, which takes 95 seconds. You change one line in `service.py` and build the image again. How long does that build take, do you think?",
        options={
            "second": "About a second, since only one line changed",
            "half": "Half of it, since Docker installs only the packages that changed",
            "right": "About 95 seconds, as long as installing the packages",
            "longer": "Longer than the first time, since Docker first has to throw the old image away",
        },
        reveal="**About 95 seconds.** The line `COPY . .` copies `service.py` too, so Docker sees a changed file at that step. It builds that step again, and every step after it, the install of the packages included, though not one package changed. With the lines in another order, the same change builds again in about a tenth of a second: the idea below shows how. The 95 seconds is an example; on your computer it depends on the packages and the network.",
    )
    guess_build
    return (guess_build,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A container.** A **container** is a program that runs with everything it needs packed in: its Python, its packages, its files and its settings, on a file system of its own. Unlike a virtual machine, a container brings no operating system of its own: it shares the kernel of the computer it runs on, the core of the operating system, which starts programs and hands out memory. So a container starts in a second and takes little memory. On a Mac or on Windows, Docker runs a small Linux virtual machine, and the containers run in it.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **An image.** A container starts from a **container image**, or **image** for short: the packed file system, with settings such as the command that starts the program. An image does not change. A container is an image that runs, with a thin layer of its own on top for what it writes; that layer is gone when the container is removed. From one image, you can start ten containers. An image holds programs for one kind of processor, such as the ARM chips of recent Macs or the Intel and AMD chips of most servers. You share images through a **registry**, such as Docker Hub: you push an image there, and a server pulls it.

        **Docker and the Dockerfile.** **Docker**, from 2013, is the best-known tool to build images and run containers; images follow an open standard, that of the Open Container Initiative, so other tools can build and run them too. A **Dockerfile** is the recipe of an image, one instruction per line. `FROM` picks the **base image** to start from, such as `python:3.12-slim`: a small Linux with Python 3.12. `WORKDIR` picks the folder to work in, `COPY` copies files from your project into the image, `RUN` runs a command, such as installing the packages, and `CMD` gives the command that starts the container. `docker build -t ops .` builds the image, and `docker run -p 8000:8000 ops` starts a container from it, with port 8000 of the computer leading to port 8000 of the container. Pin the base image like a package: `python:3.12-slim` moves along with every new release of Python 3.12, and a tag with the full version, such as `python:3.12.7-slim`, stays the same.

        **Layers and the cache.** Every `FROM`, `WORKDIR`, `COPY` and `RUN` gives a **layer**: what that one step changes in the file system. An image is its layers on top of each other, and each layer has an id. Docker keeps layers in a **build cache**. When you build again, Docker goes through the steps from the top: a step whose line is the same, and for `COPY` whose files are the same, comes from the cache. At the first step where something changed, Docker builds that step again, and every step after it, since each layer sits on the layers before it. So the order of the lines decides how quick each build is: what changes least goes at the top, such as the packages, and what changes most at the bottom, such as your own code. A file the image does not need stays out: copy only what the service uses, or list the rest in a file called `.dockerignore`.

        **Downloading an update.** A layer that is built again gets a new id, since its files are new, and so does every layer on top of it. A server that runs yesterday's image keeps its layers; of today's image, it downloads only the layers whose id it does not have yet. In order, a change to your code is a new layer of a few kilobytes; with everything copied first, the same change sends the packages along again, hundreds of megabytes.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **container** | | a program that runs with its Python, packages, files and settings packed in, on the kernel of the computer it runs on | |
        | **container image** | | the packed file system and settings a container starts from; it does not change; *image* for short | |
        | **Docker** | | the best-known tool to build images and run containers | `docker build`, `docker run` |
        | **Dockerfile** | docker-file | the recipe of an image, one instruction per line | `FROM`, `COPY`, `RUN`, `CMD` |
        | **base image** | | the image a Dockerfile starts from | `FROM python:3.12-slim` |
        | **layer** | | in an image: what one step of a Dockerfile changes in the file system | |
        | **build cache** | | the layers Docker keeps, to use again when nothing changed | |
        | **registry** | | a place to share images, such as Docker Hub | `docker push`, `docker pull` |
        """),
        L.watch_out("A secret, such as an API key, never goes in the Dockerfile or in a file you copy into the image: everyone who has the image can read every layer of it, including a layer whose file a later step deletes. Give a secret when the container starts, as a setting: `docker run -e API_KEY=… ops`."),
        L.watch_out("A container does not keep what it writes: when it is removed, its own layer goes with it. A log, or data that must stay, goes to a folder of the computer that you give at the start: `docker run -v /srv/ops-logs:/app/logs ops`."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Packing a service in a container:

    1. Start `FROM` a small base image with the Python of your project, with a tag that names the full version.
    2. Copy the files that list the packages, such as `requirements.txt`, and install them: they change least.
    3. Copy the model, then the code: what changes most goes last. When a file changes, Docker builds the first step that copies it again, and every step after it.
    4. Copy only what the service needs: logs, notes and data stay out, or go in `.dockerignore`.
    5. Give secrets, the port and folders for data when the container starts, never in the image, and build the image for the processor of the server.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The Dockerfile in order, for a service like that of the project *ops*:

    ```
    FROM python:3.12-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY model/ model/
    COPY service.py .
    CMD ["python", "service.py"]
    ```

    Here, `requirements.txt` holds the exact versions of the lock file, which `uv export` writes. You trained the model again, so the folder `model/` changed. In the cell, each step is a dictionary with its line and the files it copies; `CMD` makes no layer, so it is not in the list. The cell looks for the first step that copies a changed file. Run the cell.
    """)
    return


@app.cell
def example_first_step():
    steps = [
        {"line": "FROM python:3.12-slim", "files": []},
        {"line": "WORKDIR /app", "files": []},
        {"line": "COPY requirements.txt .", "files": ["requirements.txt"]},
        {"line": "RUN pip install -r requirements.txt", "files": []},
        {"line": "COPY model/ model/", "files": ["model/"]},
        {"line": "COPY service.py .", "files": ["service.py"]},
    ]
    changed = {"model/"}            # you trained the model again
    first = None
    for i in range(len(steps)):
        for name in steps[i]["files"]:
            if name in changed and first is None:
                first = i
    print("first step with a changed file:", first, steps[first]["line"])
    print("steps in the Dockerfile:", len(steps))
    return changed, first, i, name, steps


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Steps 0 to 3 come from the cache: their lines and files are the same as last time, so `pip install` does not run. Step 4 copies the new model and is built again, and step 5 sits on top of it, so Docker builds step 5 again too, though `service.py` did not change: two steps, half a second in this example. With a new `requirements.txt`, the first changed step is step 2, and the install runs again. Exercise 1 counts the steps for any Dockerfile and any change, and exercise 2 works out what a server then downloads.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rebuilt():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Built again

        `rebuilt(steps, changed)` gets the steps of a Dockerfile, as in the worked example, and the set of files that changed, and gives back how many steps Docker builds again. The loop goes through the numbers of the steps, `range(len(steps))`, with `i`, and through the `files` of each step with `name`. At the first `name` in `changed`, step `i` is the first step that Docker builds again. Replace the `...` with the number of steps Docker builds again from there. When no step copies a changed file, the function gives back `0`.

        `C.SLOW` is the Dockerfile of the guess, which copies everything first:

        ```
        FROM python:3.12-slim
        WORKDIR /app
        COPY . .
        RUN pip install -r requirements.txt
        CMD ["python", "service.py"]
        ```

        The check calls `rebuilt(C.QUICK, {"requirements.txt"})`: the Dockerfile of the worked example with a new list of packages. Then a new `service.py`, a new `runs.jsonl` that no step of `C.QUICK` copies, `C.SLOW` with a new `runs.jsonl`, two changed files at once, `C.SLOW` with nothing changed, a Dockerfile without steps, and a made-up Dockerfile whose second step copies two files.
        """),
        L.task("Give back how many steps Docker builds again."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rebuilt():
    box_rebuilt = L.code_box("rebuilt", lines=7)
    box_rebuilt
    return (box_rebuilt,)


@app.cell(hide_code=True)
def check_rebuilt(box_rebuilt):
    result_rebuilt = L.check("rebuilt", box_rebuilt.value)
    result_rebuilt
    return (result_rebuilt,)


@app.cell(hide_code=True)
def hints_rebuilt():
    L.hints(
        "rebuilt",
        [
            "Step 3 of the recipe: when a file changes, Docker builds the first step that copies it again, and every step after it. How many steps are there from step `i` to the last one, with `len(steps)` steps in all?",
            "In the worked example, the Dockerfile has 6 steps, and step 4 is the first changed one: Docker builds steps 4 and 5 again, 2 steps. How do you get 2 from 6 and 4?",
            "`                return len(steps) ___`",
        ],
        why="`return len(steps) - i` counts step `i` and every step after it. In `C.QUICK`, with 6 steps, a new `requirements.txt` makes step 2 the first, so Docker builds 6 − 2 = 4 steps again, the install included. A new `service.py` builds only step 5 again.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_to_download():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. What a server downloads

        Yesterday, a server pulled both images, the one of `C.SLOW` and the one of `C.QUICK`. Today, `service.py` changed, and both images were built again. `to_download(layers, have)` gets the layers of a new image and the set of layer ids the server has already, and gives back the megabytes the server still has to download. Every `layer` is a dictionary with `"id"`, `"line"`, the step it comes from, and `"mb"`, its size in megabytes. `mb` adds up the sizes, and the loop goes through the `layers` with `layer`. Replace the `...` with the question whether the server still lacks `layer`.

        The check calls `to_download(C.SLOW_IMAGE, C.ON_SERVER)` and `to_download(C.QUICK_IMAGE, C.ON_SERVER)`: today's two images, and the ids of yesterday's layers. Then an image without layers, a new server that has nothing yet, and three made-up layers of which the server has one. The sizes are examples, about what PyTorch, pandas and a small model take up.
        """),
        L.task("Add up the megabytes of the layers the server does not have yet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_to_download():
    box_to_download = L.code_box("to_download", lines=7)
    box_to_download
    return (box_to_download,)


@app.cell(hide_code=True)
def check_to_download(box_to_download):
    result_to_download = L.check("to_download", box_to_download.value)
    result_to_download
    return (result_to_download,)


@app.cell(hide_code=True)
def hints_to_download():
    L.hints(
        "to_download",
        [
            "The idea, under *Downloading an update*: which layers of a new image does a server download? Each `layer` has its id in `layer[\"id\"]`.",
            "`in` asks whether a value is in a set, as in the worked example: `\"model/\" in changed`. Python's `not in` asks the opposite.",
            "`        if layer[\"id\"] ___:`",
        ],
        why="`if layer[\"id\"] not in have:` counts only the layers the server lacks. For `C.SLOW_IMAGE`, those are the layer of `COPY . .` and the install after it: 3.05 + 380 = 383.05 MB. For `C.QUICK_IMAGE`, only the layer of `COPY service.py .`: 0.02 MB.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.md(r"""
    ### Exercise 3. What goes where?

    A container of the service needs eight things. Put each where it belongs: in the image, from the server the container runs on, or given at the start, with `docker run`.
    """)
    return


@app.cell(hide_code=True)
def sort_where_cell():
    sort_where = L.sort("where")
    sort_where
    return (sort_where,)


@app.cell(hide_code=True)
def check_where(sort_where):
    result_where = L.check("where", sort_where)
    result_where
    return (result_where,)


@app.cell(hide_code=True)
def hints_where():
    L.hints(
        "where",
        [
            "The picture of the idea, and step 5 of the recipe. Ask for each: is it the same on every server, does it belong to the computer itself, or does it differ per server or have to stay secret?",
            "Three go in the image, two come from the server, and three are given at the start.",
            "In the image: Python, the packages and \\_\\_\\_. From the server: the kernel and \\_\\_\\_. At the start: the API key, the port and \\_\\_\\_.",
        ],
        why="**In the image:** Python, the packages and the model, since the service needs them the same on every server. **From the server:** the kernel and the NVIDIA driver, since a container shares the kernel of the computer, and the driver belongs to its GPU. **Given at the start:** the API key, which must stay out of the layers; the port, which can differ per server; and the folder for the log, which must outlive the container.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_platform():
    mo.md(r"""
    ### Exercise 4. From a laptop to a server

    You build the image of the service on a laptop with an Apple M chip, push it to a registry, and pull and start it on a rented server with an Intel processor.
    """)
    return


@app.cell(hide_code=True)
def choice_platform_cell():
    choice_platform = L.choice(
        "platform",
        question="What happens?",
        options={
            "same": "It runs, since a container runs the same on every computer",
            "right": "It stops with an error, since the image holds programs for the laptop's processor",
            "rebuild": "Docker on the server builds the image again for its own processor first, and then starts it",
            "python": "It runs once Python 3.12 is installed on the server too",
        },
        correct="right",
        why={
            "right": "An image holds programs built for one kind of processor: the M chips of a Mac are ARM, and the Intel processor of the server is x86, which Docker calls amd64. Started there, the programs stop with an error such as *exec format error*. Build for the server: `docker buildx build --platform linux/amd64`, or for both kinds at once with `--platform linux/amd64,linux/arm64`. *The same on every computer* holds for computers with the same kind of processor.",
            "same": "A container brings its own Python and packages, but its programs are made for one kind of processor. Which kind is in the laptop, and which in the server?",
            "rebuild": "A server that pulls an image only gets the layers that were pushed; it never builds them again. For which processor were those layers built?",
            "python": "The image brings its own Python 3.12, so the server needs none. What else in the image depends on the computer?",
        },
    )
    choice_platform
    return (choice_platform,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Which file changed?

    Both Dockerfiles of the lesson side by side: `C.SLOW`, which copies everything first, and `C.QUICK`, in order. Choose the file that changed, and see which steps Docker builds again, how many seconds that takes in this example, and how many megabytes a server with yesterday's image downloads. The stops are 1 `runs.jsonl`, 2 `service.py`, 3 `model/` and 4 `requirements.txt`: from the file that changes most often to the one that changes least.
    """)
    return


@app.cell(hide_code=True)
def demo_change():
    L.explore("change", "the file that changed, from 1 to 4", lambda _v: C.frame(_v, L.lang), start=2,
              values=C.DEMO_VALUES, role="", task="Go through the four files: for which one is the Dockerfile in order slow too?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With everything copied first, every change costs the same: 95.5 seconds and 383.05 MB, even a change to `runs.jsonl`, which the service does not use at all. In order, `runs.jsonl` costs nothing, since no step copies it; `service.py` costs 0.1 seconds and 0.02 MB, and `model/` half a second and 3.02 MB. Only a new `requirements.txt` makes the Dockerfile in order as slow as the other: 95.6 seconds and 383.03 MB, since the packages really changed. You change your code many times a day, and your packages a few times a year.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [The same packages everywhere: environments and pinning](/course/open/sys-environments?lang=en) fixed the versions of your packages with a lock file; a container packs them with Python, the files and the start command, and its base image deserves a pinned version too. In [Someone else's computer: the cloud](/course/open/sys-cloud?lang=en), you rented a bare machine and installed the rest yourself; with a container, that machine only needs Docker. In the next lesson, *What compute costs in the cloud*, you work out what the hours of such a machine really cost, with cheaper hours on a GPU that the cloud can take back.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_platform, result_rebuilt, result_to_download, result_where):
    L.wrap_up(result_rebuilt, result_to_download, result_where, choice_platform)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
