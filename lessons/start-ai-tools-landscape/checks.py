# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson start-ai-tools-landscape, and the made-up chat that shows a context window."""
import html

from everything_ai.check import Sort

# A made-up chat in which an assistant writes a story, with a made-up length for every message in pieces of words, and
# a made-up context window of 1000 pieces, small enough for the picture. Real assistants hold far more.
WINDOW = 1000
CHAT = [
    ("you", {"en": "Write a story. Noor is afraid of water.", "nl": "Schrijf een verhaal. Noor is bang voor water."}, 20),
    ("ai", {"en": "Chapter 1: Noor moves to the coast", "nl": "Hoofdstuk 1: Noor verhuist naar de kust"}, 400),
    ("you", {"en": "Now chapter 2, about her new school.", "nl": "Nu hoofdstuk 2, over haar nieuwe school."}, 15),
    ("ai", {"en": "Chapter 2: the first day at school", "nl": "Hoofdstuk 2: de eerste schooldag"}, 420),
    ("you", {"en": "Chapter 3: a storm.", "nl": "Hoofdstuk 3: een storm."}, 10),
    ("ai", {"en": "Chapter 3: the night of the storm", "nl": "Hoofdstuk 3: de nacht van de storm"}, 450),
    ("you", {"en": "And now the summer vacation.", "nl": "En nu de zomervakantie."}, 8),
    ("ai", {"en": "Chapter 4: Noor dives into the sea", "nl": "Hoofdstuk 4: Noor duikt de zee in"}, 430),
]
_WHO = {"you": {"en": "You", "nl": "Jij"}, "ai": {"en": "Assistant", "nl": "Assistent"}}


def first_seen(n: int) -> int:
    """The index of the oldest message the model still sees when the chat has its first n messages: counting back from
    the newest, as many messages as fit in the window together."""
    total, first = 0, n
    for i in range(n - 1, -1, -1):
        if total + CHAT[i][2] > WINDOW:
            break
        total += CHAT[i][2]
        first = i
    return first


def _grouped(x: int, lang: str) -> str:
    return f"{x:,}".replace(",", "." if lang == "nl" else ",")


def window_frame(n: int, lang: str) -> tuple[str, str]:
    """The chat with its first n messages, and a bracket beside the newest ones that fit in the context window; the
    older ones are grey. Then the sentence under the picture."""
    n = int(n)
    first = first_seen(n)
    W, ROW, GAP, TOP = 660, 30, 6, 4
    longest = max(size for _, _, size in CHAT)
    parts = []
    for i, (who, text, size) in enumerate(CHAT[:n]):
        y = TOP + i * (ROW + GAP)
        inside = i >= first
        ink = "var(--ea-ink)" if inside else "var(--ea-ink-3)"
        bar = 4 + 120 * size / longest
        parts.append(
            f'<text x="0" y="{y + 20}" style="font-size:12.5px;font-weight:600;fill:{ink}">{_WHO[who][lang]}</text>'
            f'<text x="78" y="{y + 20}" style="font-size:13px;fill:{ink}">{html.escape(text[lang])}</text>'
            f'<rect x="368" y="{y + 8}" width="{bar:.1f}" height="14" rx="3" style="fill:{ink};fill-opacity:{0.55 if inside else 0.25}"/>'
            f'<text x="{368 + bar + 6:.1f}" y="{y + 20}" style="font-size:12px;fill:{ink};font-variant-numeric:tabular-nums">{size}</text>')
    y0, y1 = TOP + first * (ROW + GAP), TOP + n * (ROW + GAP) - GAP
    window = {"en": "context window", "nl": "context window"}[lang]
    pieces = {"en": f"{_grouped(WINDOW, lang)} pieces", "nl": f"{_grouped(WINDOW, lang)} stukjes"}[lang]
    parts.append(f'<path d="M532 {y0} H540 V{y1} H532" style="fill:none;stroke:var(--ea-ink);stroke-width:1.5"/>'
                 f'<text x="548" y="{(y0 + y1) / 2 - 2:.1f}" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">{window}</text>'
                 f'<text x="548" y="{(y0 + y1) / 2 + 14:.1f}" style="font-size:12px;fill:var(--ea-ink-2)">{pieces}</text>')
    if first > 0:
        gone = {"en": "no longer seen", "nl": "niet meer gezien"}[lang]
        parts.append(f'<text x="548" y="{TOP + first * (ROW + GAP) / 2 + 4:.1f}" style="font-size:12px;fill:var(--ea-ink-3)">{gone}</text>')
    H = TOP + max(n, 1) * (ROW + GAP)
    title = {"en": "A chat and its context window", "nl": "Een chat en zijn context window"}[lang]
    svg = (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:40rem">'
           f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
           f'<title>{title}</title>{"".join(parts)}</svg></figure>')
    total = sum(size for _, _, size in CHAT[:n])
    seen = sum(size for _, _, size in CHAT[first:n])
    if lang == "nl":
        said = (f"{n} {'bericht' if n == 1 else 'berichten'}, samen {_grouped(total, lang)} stukjes. "
                + (f"Alles past in het context window. " if first == 0 else
                   f"Het context window houdt de nieuwste {n - first}: {seen} stukjes. ")
                + ("Bericht 1, waarin staat dat Noor bang is voor water, ziet het model nog."
                   if first == 0 else "Bericht 1, waarin staat dat Noor bang is voor water, ziet het model niet meer."))
    else:
        said = (f"{n} {'message' if n == 1 else 'messages'}, {_grouped(total, lang)} pieces in all. "
                + (f"Everything fits in the context window. " if first == 0 else
                   f"The context window holds the newest {n - first}: {seen} pieces. ")
                + ("The model still sees message 1, which says that Noor is afraid of water."
                   if first == 0 else "The model no longer sees message 1, which says that Noor is afraid of water."))
    return svg, said


# The lesson is done when these are correct. "which-tool" and "long-chat" are choice questions in the lesson.
REQUIRED = ["before-you-paste", "which-tool", "long-chat"]

_PRIVATE = {"en": "Who else is this about, or who could misuse it? Then it is not for a chat on the web as it is.",
            "nl": "Over wie gaat dit nog meer, of wie kan er misbruik van maken? Dan hoort het niet zomaar in een chat op het web."}
_FINE = {"en": "Is there anything in it about you or someone else that is not for everyone?",
         "nl": "Staat er iets in over jou of iemand anders dat niet voor iedereen is?"}

EXERCISES = {
    "before-you-paste": Sort(
        name="before-you-paste",
        groups={
            "paste": {"en": "Paste as it is", "nl": "Plak zoals het is"},
            "first": {"en": "Not as it is", "nl": "Niet zoals het is"},
        },
        items={
            "pancakes": {"en": "**Pancakes.** A pancake recipe from a cookbook, to make it for eight people.",
                         "nl": "**Pannenkoeken.** Een recept voor pannenkoeken uit een kookboek, om het voor acht mensen te maken."},
            "password": {"en": "**Password.** Your email password, to hear whether it is strong enough.",
                         "nl": "**Wachtwoord.** Het wachtwoord van je mail, om te horen of het sterk genoeg is."},
            "letter": {"en": "**Letter.** A letter from the doctor with your neighbor's name and illness, to explain the medical words to them.",
                       "nl": "**Brief.** Een brief van de dokter met de naam en de ziekte van je buur, om je buur de medische woorden uit te leggen."},
            "news": {"en": "**News.** An article from a newspaper's website, to sum it up in three sentences.",
                     "nl": "**Nieuws.** Een stuk van de website van een krant, om het in drie zinnen samen te vatten."},
            "customers": {"en": "**Customers.** A list from work with the names and phone numbers of customers, to sort it.",
                          "nl": "**Klanten.** Een lijst van je werk met de namen en telefoonnummers van klanten, om hem te sorteren."},
            "wiki": {"en": "**Wikipedia.** A paragraph from Wikipedia about windmills, to have it explained more simply.",
                     "nl": "**Wikipedia.** Een alinea van Wikipedia over windmolens, om hem eenvoudiger uitgelegd te krijgen."},
        },
        solution={"pancakes": "paste", "password": "first", "letter": "first", "news": "paste", "customers": "first", "wiki": "paste"},
        nudge={"password": _PRIVATE, "letter": _PRIVATE, "customers": _PRIVATE, "pancakes": _FINE, "news": _FINE, "wiki": _FINE},
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the six parts again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zes delen opnieuw uit."},
            "empty": {"en": "Pick a group for each of the six.",
                      "nl": "Kies voor elk van de zes een groep."},
            "unfinished": {"en": "Everything you picked so far is right. Pick a group for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest een groep (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are right. The password never goes into a chat; the letter can go once the name and other details that point to your neighbor are out; the customer list stays at work, or goes into a tool your employer approved.",
                        "nl": "Alle {n} kloppen. Het wachtwoord gaat nooit een chat in; de brief kan zonder de naam en de andere gegevens die naar je buur wijzen; de klantenlijst blijft op je werk, of gaat in een tool die je werkgever heeft goedgekeurd."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"pancakes": "paste", "password": "first", "letter": "first", "news": "paste", "customers": "first", "wiki": "paste"}
CASES = [
    ("before-you-paste", {}, "empty", None),
    ("before-you-paste", _RIGHT, "correct", None),
    ("before-you-paste", {"pancakes": "paste", "password": "first"}, "wrong", "unfinished"),
    ("before-you-paste", {**_RIGHT, "letter": "paste"}, "wrong", "wrong:letter"),
    ("before-you-paste", {**_RIGHT, "wiki": "first"}, "wrong", "wrong:wiki"),
    ("before-you-paste", {**_RIGHT, "customers": "paste", "news": "first"}, "wrong", "wrong:news"),
]
