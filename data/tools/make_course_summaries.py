# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-summaries.json: the documents the lessons on RAG of the domain bouwen search in. Every point of the
summary of every built lesson (`summary` in lessons/<id>/lesson.yaml), in English and Dutch, with its lesson, the
lesson's title, and its domain, module and level from the route; and the questions those lessons ask about them, each
with the point that answers it (`gold`) and a pattern that a right answer matches (`answer`), or none of both for a
question the points do not answer.

The file is a snapshot: lessons built later do not change it, so the numbers the lessons state stay the same. Run it
again only on purpose, from the course folder:

    uv run python data/tools/make_course_summaries.py
"""
import json
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# kind "words": the question shares words with its point; "meaning": it asks the same in other words; "none": no point
# answers it. A pattern is a regular expression for a right answer, matched on the answer in lower case.
QUESTIONS = [
    ("words", "cnn-pytorch#2", r"9[,.]?098",
     "How many parameters does the small CNN of the course have?",
     "Hoeveel parameters heeft het kleine CNN van de course?"),
    ("words", "lm-perplexity#1", r"11[.,]6",
     "What is the perplexity of the counted bigram model on Tiny Shakespeare?",
     "Wat is de perplexity van het getelde bigram-model op Tiny Shakespeare?"),
    ("words", "cv-alexnet#0", r"15[.,]3",
     "What top-5 error did AlexNet reach in 2012?",
     "Welke top-5 error haalde AlexNet in 2012?"),
    ("words", "torch-pretrained#2", r"\b929\b",
     "How many of the 1,000 test digits does the downloaded digit reader read right?",
     "Hoeveel van de 1.000 testcijfers leest de gedownloade cijferlezer goed?"),
    ("words", "build-project-call#0", r"\b0\b|\bnone\b|\bzero\b|\bnever\b|\bgeen\b|\bnul\b|\bnooit\b",
     "In how many of 8 runs did Qwen3 name Sam without the earlier turn in the request?",
     "In hoeveel van de 8 runs noemde Qwen3 Sam zonder de eerdere beurt in het request?"),
    ("words", "llm-data-pipeline#0", r"\b17\b",
     "How many GB are the ids of nanoGPT's OpenWebText?",
     "Hoeveel GB zijn de ids van OpenWebText van nanoGPT?"),
    ("words", "tok-bpe#2", r"50[,.]?000",
     "How many rules did GPT-2 learn for its tokens?",
     "Hoeveel regels leerde GPT-2 voor zijn tokens?"),
    ("words", "ml-recommend-problem#2", r"21[.,]7",
     "For what share of the customers does the popularity baseline find something at k = 10?",
     "Voor welk deel van de klanten vindt de popularity baseline iets bij k = 10?"),
    ("words", "ml-pca#2", r"\b301\b",
     "On two components of PCA, how many of the 342 penguins does k-means put in the cluster of their own species?",
     "Op twee componenten van PCA, hoeveel van de 342 pinguïns zet k-means in het cluster van hun eigen soort?"),
    ("words", "cv-color-cnn#0", r"\b224\b",
     "How many parameters does nn.Conv2d(3, 8, 3) have?",
     "Hoeveel parameters heeft nn.Conv2d(3, 8, 3)?"),
    ("words", "build-prompt-caching#1", r"4[.,]78",
     "How many seconds did a request take without the cache?",
     "Hoeveel seconden duurde een request zonder de cache?"),
    ("words", "build-defensive-prompts#1", r"\b17\b",
     "When Qwen3 sorted the emails the plain way, how many of the 24 answers followed the email?",
     "Toen Qwen3 de e-mails op de gewone manier sorteerde, hoeveel van de 24 antwoorden volgden de e-mail?"),
    ("words", "ml-dbscan#1", r"\b20\b",
     "How many strays are there among the 400 made-up points that DBSCAN clusters?",
     "Hoeveel losse punten zitten er tussen de 400 verzonnen punten die DBSCAN clustert?"),
    ("words", "cv-morphology#2", r"1[,.]?331",
     "How many specks does an opening of 3 by 3 remove from the apples?",
     "Hoeveel spikkels haalt een opening van 3 bij 3 weg van de appels?"),
    ("words", "cv-grad-cam#2", r"\b95\b",
     "What share of the Grad-CAM of Egyptian cat lies on the cat half of the collage?",
     "Welk deel van de Grad-CAM van Egyptian cat ligt op de helft met de kat van de collage?"),
    ("words", "llm-pretrain-capstone#1", r"\b20\b",
     "How many tokens per parameter does compute_optimal use?",
     "Hoeveel tokens per parameter gebruikt compute_optimal?"),
    ("words", "res-share-work#2", r"\b64\b(?![ -]?(?:bit|byte))",
     "How many hex digits does a SHA-256 checksum have?",
     "Hoeveel hextekens heeft een SHA-256-checksum?"),
    ("meaning", "llm-scaling-laws#0", r"straight|power|recht|macht",
     "What shape does the curve of a model's loss against its size take with both axes on a logarithmic scale?",
     "Welke vorm heeft de kromme van de loss van een model tegen zijn grootte als beide assen logaritmisch zijn?"),
    ("meaning", "sys-memory-accounting#0", r"gradient",
     "Apart from the weights themselves, what else has to fit in memory while a network learns?",
     "Wat moet er naast de weights zelf nog meer in het geheugen passen terwijl een netwerk leert?"),
    ("meaning", "rl-what-is-rl#0", r"reward|beloning",
     "What tells an agent how good the step it just took was?",
     "Wat vertelt een agent hoe goed de stap was die hij net zette?"),
    ("meaning", "ml-knn#0", r"vot|majority|meerderheid|stemm",
     "How does a classifier that looks at the closest stored examples pick a class for something new?",
     "Hoe kiest een classifier die naar de dichtstbijzijnde bewaarde voorbeelden kijkt een klasse voor iets nieuws?"),
    ("meaning", "stats-bootstrap#0", r"\bfour\b|\b4\b|\bvier\b|quadruple|verviervoudig",
     "How much more data do you need to halve the uncertainty of an average?",
     "Hoeveel meer data heb je nodig om de onzekerheid van een gemiddelde te halveren?"),
    ("meaning", "build-tokens-prices#1", r"\bmore\b|\bmeer\b",
     "Does a sentence in Dutch need more or fewer pieces than the same sentence in English?",
     "Heeft een zin in het Nederlands meer of minder stukjes nodig dan dezelfde zin in het Engels?"),
    ("meaning", "sys-number-formats#2", r"lost|verlo|unchanged|onveranderd|stays 1|blijft 1|nothing|niets",
     "What happens when you add a thousandth to one in bfloat16?",
     "Wat gebeurt er als je in bfloat16 een duizendste bij één optelt?"),
    ("none", None, None,
     "How much does the course cost per month?",
     "Hoeveel kost de course per maand?"),
    ("none", None, None,
     "Which graphics card did the course record its runs on?",
     "Op welke videokaart nam de course zijn runs op?"),
    ("none", None, None,
     "In which city is the bike shop of the handbook?",
     "In welke stad staat de fietsenwinkel van het handboek?"),
    ("none", None, None,
     "How many people have finished the course?",
     "Hoeveel mensen hebben de course afgemaakt?"),
]


def route() -> dict:
    out = {}
    for f in sorted((ROOT / "course" / "route" / "lessen").glob("*.yaml")):
        for lesson in yaml.safe_load(f.read_text(encoding="utf-8"))["lessons"]:
            out[lesson["id"]] = lesson
    return out


def main() -> None:
    where = route()
    points = []
    for f in sorted((ROOT / "lessons").glob("*/lesson.yaml")):
        meta = yaml.safe_load(f.read_text(encoding="utf-8"))
        r = where[meta["id"]]
        for i, s in enumerate(meta.get("summary") or []):
            points.append({"id": f'{meta["id"]}#{i}', "lesson": meta["id"], "title": meta["title"],
                           "domain": r.get("domein"), "module": r.get("module"), "level": r.get("level"),
                           "text": {"en": s["en"], "nl": s["nl"]}})
    ids = {p["id"] for p in points}
    questions = []
    for n, (kind, gold, answer, en, nl) in enumerate(QUESTIONS, start=1):
        assert (gold is None) == (kind == "none") and (gold is None or gold in ids), gold
        if gold is not None:   # the point gives the answer the pattern looks for, in both languages
            text = next(p["text"] for p in points if p["id"] == gold)
            assert all(re.search(answer, text[lang].lower()) for lang in ("en", "nl")) or kind == "meaning", gold
        questions.append({"id": f"q{n:02d}", "kind": kind, "gold": gold, "answer": answer, "text": {"en": en, "nl": nl}})
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {"made_from": commit, "lessons": len({p["lesson"] for p in points}), "points": points, "questions": questions}
    (ROOT / "data" / "course-summaries.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(len(points), "points from", out["lessons"], "lessons,", len(questions), "questions")


if __name__ == "__main__":
    main()
