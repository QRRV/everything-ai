# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-classes: a counter and a playlist, each a class written in a box and tried in scenarios."""
import re

from everything_ai.check import ClassBox, Mistake

# A `...` that is still there, found before the code runs: a method that is still `...` gives back nothing,
# and that should not be explained as some other mistake.
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your class. Replace each one with your own code.",
                nl="Er staat nog een `...` in je class. Vervang ze allemaal door je eigen code.")

_SYNTAX = [
    Mistake("no-self", syntax=r"^\s+def\s+\w+\(\s*(\)|(?!self\b)\w)",
            en="Every method gets the object itself as its first argument: write `self` first between the parentheses, as in `def add(self):`.",
            nl="Elke method krijgt het object zelf als eerste argument: zet `self` als eerste tussen de haakjes, zoals in `def add(self):`."),
    Mistake("init-spelling", syntax=r"def\s+(?!__init__\s*\()(_{0,3}in{1,2}it_{0,3}|__int__)\s*\(",
            en="The method that sets the starting data is called `__init__`, with two underscores on each side.",
            nl="De method die de begindata zet, heet `__init__`, met twee lage streepjes aan elke kant."),
    Mistake("colon-missing", syntax=r"^[ \t]*(class\b[^:\n]*|def\b[^:\n]*|for\b[^:\n]*)$",
            en="The line `{line}` must end with a colon, `:`. The colon tells Python that the indented lines below belong to it.",
            nl="De regel `{line}` moet eindigen op een dubbele punt, `:`. Die vertelt Python dat de ingesprongen regels eronder erbij horen."),
]

# The lesson is done when these are correct. "two-objects" is the choice question in the lesson.
REQUIRED = ["counter", "two-objects", "playlist"]

_COUNTER_START = "class Counter:\n    def __init__(self):\n        self.count = ...\n\n    def add(self):\n        ..."
_COUNTER = "class Counter:\n    def __init__(self):\n        self.count = 0\n\n    def add(self):\n        self.count = self.count + 1"
_PLAYLIST_START = ("class Playlist:\n    def __init__(self, name):\n        self.name = name\n        self.songs = ...\n\n"
                   "    def add(self, song):\n        ...\n\n    def length(self):\n        ...")
_PLAYLIST = ("class Playlist:\n    def __init__(self, name):\n        self.name = name\n        self.songs = []\n\n"
             "    def add(self, song):\n        self.songs.append(song)\n\n    def length(self):\n        return len(self.songs)")
_PLAYLIST_SHARED = ("class Playlist:\n    songs = []\n\n    def __init__(self, name):\n        self.name = name\n\n"
                    "    def add(self, song):\n        self.songs.append(song)\n\n    def length(self):\n        return len(self.songs)")

EXERCISES = {
    "counter": ClassBox(
        name="counter",
        start=_COUNTER_START,
        solution=_COUNTER,
        scenarios=[
            "c = Counter()\nc.add()\nc.add()\nc.count",
            "c = Counter()\nc.count",
            "a = Counter()\nb = Counter()\na.add()\nb.count",
            "c = Counter()\nfor i in range(5):\n    c.add()\nc.count",
        ],
        first=[_DOTS],
        mistakes=_SYNTAX + [
            Mistake("local-count", syntax=r"^\s+count\s*\+?=|=\s*count\b",
                    en="Inside a method, `count` without `self.` in front is a new name that disappears when the method ends. The object's own count is `self.count`: write `self.count = self.count + 1`.",
                    nl="In een method is `count` zonder `self.` ervoor een nieuwe naam, die verdwijnt als de method klaar is. De eigen telling van het object is `self.count`: schrijf `self.count = self.count + 1`."),
            Mistake("always-one", wrong=_COUNTER.replace("self.count = self.count + 1", "self.count = 1"),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyour counter is `{got}`, and it should be `{want}`. `add` sets the count to 1 each time. Let it add 1 to what `self.count` already is.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nis je teller `{got}`, en dat moet `{want}` zijn. `add` zet de telling elke keer op 1. Laat hem 1 optellen bij wat `self.count` al is."),
            Mistake("start-one", wrong=_COUNTER.replace("self.count = 0", "self.count = 1"),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyour counter is `{got}`, and it should be `{want}`: one too many. A new counter has counted nothing yet, so it starts at 0.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nis je teller `{got}`, en dat moet `{want}` zijn: één te veel. Een nieuwe teller heeft nog niets geteld, dus begint hij bij 0."),
            Mistake("not-stored", wrong=_COUNTER.replace("self.count = self.count + 1", "self.count + 1"),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyour counter is `{got}`, and it should be `{want}`. `add` works out `self.count + 1`, but does not store it, so the count stays the same. Store it with `=`: `self.count = self.count + 1`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nis je teller `{got}`, en dat moet `{want}` zijn. `add` rekent `self.count + 1` uit, maar bewaart het niet, dus blijft de telling gelijk. Bewaar het met `=`: `self.count = self.count + 1`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the starting count in `__init__`, and in `add`, the line that adds 1 to it.",
                      "nl": "Vervang de twee `...`: de begintelling in `__init__`, en in `add` de regel die er 1 bij optelt."},
            "cleared": {"en": "There is no code in the box any more. Start again with `class Counter:`, then the two methods, `__init__` and `add`.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `class Counter:`, en dan de twee methods, `__init__` en `add`."},
            "wrong": {"en": "After these lines:\n\n```python\n{scenario}\n```\n\nyour counter is `{got}`, and it should be `{want}`. Does `__init__` start `self.count` at 0, and does `add` add 1 to it?",
                      "nl": "Na deze regels:\n\n```python\n{scenario}\n```\n\nis je teller `{got}`, en dat moet `{want}` zijn. Begint `self.count` in `__init__` bij 0, en telt `add` er 1 bij op?"},
            "almost": {"en": "Right for the example. But after these lines:\n\n```python\n{scenario}\n```\n\nyour counter is `{got}`, and it should be `{want}`. Does each object keep its own count in `self.count`?",
                       "nl": "Goed voor het voorbeeld. Maar na deze regels:\n\n```python\n{scenario}\n```\n\nis je teller `{got}`, en dat moet `{want}` zijn. Houdt elk object zijn eigen telling bij in `self.count`?"},
            "correct": {"en": "Your counter is right in all {n} cases: it starts at 0, counts up, and two counters each keep their own count.",
                        "nl": "Je teller klopt in alle {n} gevallen: hij begint bij 0, telt op, en twee tellers houden elk hun eigen telling bij."},
        },
    ),
    "playlist": ClassBox(
        name="playlist",
        start=_PLAYLIST_START,
        solution=_PLAYLIST,
        scenarios=[
            "p = Playlist(\"Run\")\np.add(\"Intro\")\np.add(\"Rain\")\np.length()",
            "p = Playlist(\"Run\")\np.add(\"Intro\")\np.add(\"Rain\")\np.songs",
            "p = Playlist(\"Quiet\")\np.length()",
            "p = Playlist(\"Quiet\")\np.name",
            "a = Playlist(\"A\")\nb = Playlist(\"B\")\na.add(\"Rain\")\nb.length()",
        ],
        first=[_DOTS],
        mistakes=_SYNTAX + [
            Mistake("local-songs", syntax=r"len\(\s*songs\s*\)|^\s+songs\s*\.\s*append|^\s+songs\s*=",
                    en="Inside a method, `songs` without `self.` in front is not the object's list. The object's own list is `self.songs`.",
                    nl="In een method is `songs` zonder `self.` ervoor niet de lijst van het object. De eigen lijst van het object is `self.songs`."),
            Mistake("append-assign", syntax=r"=\s*self\s*\.\s*songs\s*\.\s*append\s*\(",
                    en="`append` changes the list itself and gives back nothing, so `self.songs = self.songs.append(song)` puts nothing, `None`, in `self.songs`. Write only `self.songs.append(song)`.",
                    nl="`append` verandert de lijst zelf en geeft niets terug, dus zet `self.songs = self.songs.append(song)` niets, `None`, in `self.songs`. Schrijf alleen `self.songs.append(song)`."),
            Mistake("shared-list", state="almost", wrong=_PLAYLIST_SHARED,
                    en="Your playlists share one list: `songs = []` is written in the class itself, not in `__init__`, so every playlist gets the same list. Give each playlist its own list in `__init__`: `self.songs = []`.",
                    nl="Je playlists delen één lijst: `songs = []` staat in de class zelf, niet in `__init__`, dus krijgt elke playlist dezelfde lijst. Geef elke playlist zijn eigen lijst in `__init__`: `self.songs = []`."),
            Mistake("print-length", matches=lambda got, text: got is None and re.search(r"def\s+length\s*\([^)]*\)\s*:\s*\n\s+print\s*\(", text) is not None,
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, Python's word for nothing: `length` shows the number with `print`, but gives nothing back. Use `return`.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, het woord van Python voor niets: `length` laat het getal zien met `print`, maar geeft niets terug. Gebruik `return`."),
            Mistake("gives-nothing", matches=lambda got, text: got is None,
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, Python's word for nothing. Does the method that should give something back have a `return`?",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, het woord van Python voor niets. Heeft de method die iets terug moet geven een `return`?"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the three `...`: an empty list for `self.songs`, the line in `add` that adds `song` to it, and the line in `length` that gives back its length.",
                      "nl": "Vervang de drie `...`: een lege lijst voor `self.songs`, de regel in `add` die `song` eraan toevoegt, en de regel in `length` die de lengte ervan teruggeeft."},
            "cleared": {"en": "There is no code in the box any more. Start again with `class Playlist:`, then the methods `__init__`, `add` and `length`.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `class Playlist:`, en dan de methods `__init__`, `add` en `length`."},
            "wrong": {"en": "After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does `add` put the song in `self.songs`, and does `length` give back `len(self.songs)`?",
                      "nl": "Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Zet `add` het nummer in `self.songs`, en geeft `length` `len(self.songs)` terug?"},
            "almost": {"en": "Right for the example. But after these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does each playlist keep its own name and its own list of songs, and do `add` and `length` use that list?",
                       "nl": "Goed voor het voorbeeld. Maar na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Houdt elke playlist een eigen naam en een eigen lijst met nummers bij, en gebruiken `add` en `length` die lijst?"},
            "correct": {"en": "Your playlist is right in all {n} cases: songs go in, its length is right, and two playlists each keep their own songs.",
                        "nl": "Je playlist klopt in alle {n} gevallen: nummers gaan erin, de lengte klopt, en twee playlists houden elk hun eigen nummers bij."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("counter", _COUNTER_START, "empty", None),
    ("counter", "", "empty", None),
    ("counter", _COUNTER, "correct", None),
    ("counter", _COUNTER.replace("self.count = self.count + 1", "self.count += 1"), "correct", None),
    ("counter", "class Counter:\n    count = 0\n\n    def add(self):\n        self.count += 1", "correct", None),
    ("counter", _COUNTER.replace("self.count = self.count + 1", "self.count = 1"), "wrong", "always-one"),
    ("counter", _COUNTER.replace("self.count = 0", "self.count = 1"), "wrong", "start-one"),
    ("counter", _COUNTER.replace("self.count = self.count + 1", "self.count + 1"), "wrong", "not-stored"),
    ("counter", _COUNTER.replace("self.count = self.count + 1", "return self.count + 1"), "wrong", "not-stored"),
    ("counter", _COUNTER.replace("self.count = self.count + 1", "..."), "wrong", "dots-left"),
    ("counter", _COUNTER.replace("self.count = self.count + 1", "count = count + 1"), "error", "local-count"),
    ("counter", _COUNTER.replace("self.count = self.count + 1", "count += 1"), "error", "local-count"),
    ("counter", _COUNTER.replace("self.count = self.count + 1", "self.count = count + 1"), "error", "local-count"),
    ("counter", _COUNTER.replace("def add(self):", "def add():"), "error", "no-self"),
    ("counter", _COUNTER.replace("__init__", "_init_"), "error", "init-spelling"),
    ("counter", _COUNTER.replace("__init__", "__init"), "error", "init-spelling"),
    ("counter", _COUNTER.replace("__init__", "__innit__"), "error", "init-spelling"),
    ("counter", _COUNTER.replace("class Counter:", "class Counter"), "error", "colon-missing"),
    ("counter", _COUNTER.replace("class Counter:", "class counter:"), "error", None),
    ("counter", _COUNTER.replace("def add(self):", "def plus(self):"), "error", None),
    ("playlist", _PLAYLIST_START, "empty", None),
    ("playlist", _PLAYLIST, "correct", None),
    ("playlist", _PLAYLIST.replace("self.songs.append(song)", "self.songs = self.songs + [song]"), "correct", None),
    ("playlist", _PLAYLIST.replace("return len(self.songs)", "return len(songs)"), "error", "local-songs"),
    ("playlist", _PLAYLIST.replace("self.songs = []", "songs = []"), "error", "local-songs"),
    ("playlist", _PLAYLIST.replace("self.songs.append(song)", "self.songs = self.songs.append(song)"), "error", "append-assign"),
    ("playlist", _PLAYLIST.replace("def add(self, song):", "def add(song):"), "error", "no-self"),
    ("playlist", _PLAYLIST.replace("return len(self.songs)", "print(len(self.songs))"), "wrong", "print-length"),
    ("playlist", _PLAYLIST.replace("return len(self.songs)", "len(self.songs)"), "wrong", "gives-nothing"),
    ("playlist", _PLAYLIST_SHARED, "almost", "shared-list"),
    ("playlist", _PLAYLIST.replace("self.songs.append(song)", "self.songs = song"), "wrong", None),
    ("playlist", _PLAYLIST.replace("        self.name = name\n", ""), "error", None),
]
