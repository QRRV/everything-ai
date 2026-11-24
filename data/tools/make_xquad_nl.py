# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/xquad-squad-nl.json.gz: the 2,067 paragraphs of the 48 articles of English Wikipedia in the development
set of SQuAD 1.1, in English and in Dutch, with the questions and answers of the 240 paragraphs of XQuAD, for the
lessons of domain taal about topics, questions and search (read with everything_ai.xquad).

SQuAD 1.1 (Rajpurkar, Zhang, Lopyrev and Liang, EMNLP 2016; CC-BY-SA-4.0) asked crowd workers questions about
paragraphs of Wikipedia, each answered by a stretch of its paragraph; its development set has 2,067 paragraphs of 48
articles with 10,570 questions, most with three answers by different people. XQuAD (Artetxe, Ruder and Yogatama, ACL
2020; CC-BY-SA-4.0) took 240 of those paragraphs, five per article, with their 1,190 questions. SQuAD-NL v1.1 (de Vries,
Wieling and Nissim, DUMB, EMNLP 2023; CC-BY-SA-4.0) translated SQuAD into Dutch with Google Translate; people
corrected the 240 paragraphs of XQuAD and their questions (its test set, 1,183 questions), the other paragraphs are
machine translation (its dev set). The file keeps every paragraph in both languages, and the questions with all their
answers only for the 240 of XQuAD (`xquad`). The texts are unchanged. The course adds a Dutch name for each article
(TITLES_NL) and `shown`, 0 for the paragraphs that a lesson does not show as an example (one with a violent or sexual
word, found with a word list, WORDS); they stay in the file.

SQuAD and SQuAD-NL come from Hugging Face as parquet files at a fixed commit of rajpurkar/squad and of
GroNLP/squad-nl-v1.1. Download the three files below into one folder and run, from the course folder:

    uv run --with pyarrow python data/tools/make_xquad_nl.py <folder>
"""
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

SQUAD_COMMIT = "7b6d24c440a36b6815f21b70d25016731768db1f"
NL_COMMIT = "7e1167c6ba074b3aedcd11061b0a921d5383106d"
SOURCES = {
    "validation-00000-of-00001.parquet": (
        f"https://huggingface.co/datasets/rajpurkar/squad/resolve/{SQUAD_COMMIT}/plain_text/"
        "validation-00000-of-00001.parquet", "8c6646d36bd5a95061e076788cf3161d11f6f3e7d625dac7a83bbed0a49f69f7"),
    "dev-00000-of-00001.parquet": (
        f"https://huggingface.co/datasets/GroNLP/squad-nl-v1.1/resolve/{NL_COMMIT}/data/dev-00000-of-00001.parquet",
        "99a10e8d55f4fdca27ba098158caf51be4619c6a320c5a72e7f4356b6ee2bd8a"),
    "test-00000-of-00001.parquet": (
        f"https://huggingface.co/datasets/GroNLP/squad-nl-v1.1/resolve/{NL_COMMIT}/data/test-00000-of-00001.parquet",
        "64682ff1c2b1c4c656700dd10f9606502c6572b0d7799e0f146dfe6c09e8adab"),
}
OUT = Path(__file__).resolve().parents[2] / "data" / "xquad-squad-nl.json.gz"
TITLES_NL = {   # the course's Dutch name for each article
    "Super_Bowl_50": "Super Bowl 50", "Warsaw": "Warschau", "Normans": "Normandiërs", "Nikola_Tesla": "Nikola Tesla",
    "Computational_complexity_theory": "Complexiteitstheorie", "Teacher": "Leraar", "Martin_Luther": "Maarten Luther",
    "Southern_California": "Zuid-Californië", "Sky_(United_Kingdom)": "Sky (Verenigd Koninkrijk)",
    "Victoria_(Australia)": "Victoria (Australië)", "Huguenot": "Hugenoten", "Steam_engine": "Stoommachine",
    "Oxygen": "Zuurstof", "1973_oil_crisis": "Oliecrisis van 1973", "Apollo_program": "Apolloprogramma",
    "European_Union_law": "Recht van de Europese Unie", "Amazon_rainforest": "Amazoneregenwoud",
    "Ctenophora": "Ribkwallen", "Fresno,_California": "Fresno (Californië)", "Packet_switching": "Pakketschakeling",
    "Black_Death": "Zwarte Dood", "Geology": "Geologie", "Newcastle_upon_Tyne": "Newcastle upon Tyne",
    "Victoria_and_Albert_Museum": "Victoria and Albert Museum",
    "American_Broadcasting_Company": "American Broadcasting Company", "Genghis_Khan": "Dzjengis Khan",
    "Pharmacy": "Farmacie", "Immune_system": "Immuunsysteem", "Civil_disobedience": "Burgerlijke ongehoorzaamheid",
    "Construction": "Bouw", "Private_school": "Particuliere school", "Harvard_University": "Harvard-universiteit",
    "Jacksonville,_Florida": "Jacksonville (Florida)", "Economic_inequality": "Economische ongelijkheid",
    "Doctor_Who": "Doctor Who", "University_of_Chicago": "Universiteit van Chicago", "Yuan_dynasty": "Yuan-dynastie",
    "Kenya": "Kenia", "Intergovernmental_Panel_on_Climate_Change": "Intergovernmental Panel on Climate Change",
    "Chloroplast": "Bladgroenkorrel", "Prime_number": "Priemgetal", "Rhine": "Rijn",
    "Scottish_Parliament": "Schots Parlement", "Islamism": "Islamisme", "Imperialism": "Imperialisme",
    "United_Methodist_Church": "United Methodist Church", "French_and_Indian_War": "Franse en Indiaanse Oorlog",
    "Force": "Kracht",
}
# words that keep a paragraph from being shown as an example, in English and in Dutch
WORDS = re.compile(r"\b(kill\w*|murder\w*|massacr\w*|slaughter\w*|rape\w*|tortur\w*|terror\w*|suicid\w*|bomb\w*|"
                   r"genocid\w*|execution\w*|executed|behead\w*|sex\w*|nude\w*|naked|slave\w*|"
                   r"doodde|doodden|gedood|vermoord\w*|moord\w*|bloedbad\w*|afgeslacht\w*|verkracht\w*|marteld\w*|"
                   r"marteling\w*|zelfmoord\w*|bom|bommen|bombard\w*|genocide|executie\w*|geëxecuteerd|onthoofd\w*|"
                   r"seks\w*|naakt\w*|slaaf|slaven\w*|slavernij)\b", re.IGNORECASE)


def _table(folder: Path, name: str) -> dict:
    import pyarrow.parquet as pq

    path = folder / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCES[name][1], f"{path} is not the file of the commit"
    return pq.read_table(path).to_pydict()


def _answers(context: str, answers: dict) -> list:
    """[text, start] for every answer, each checked to stand at its start in the paragraph."""
    out = []
    for text, start in zip(answers["text"], answers["answer_start"]):
        assert context[start:start + len(text)] == text, (text, start)
        out.append([text, int(start)])
    return out


def articles(folder: Path) -> list:
    en = _table(folder, "validation-00000-of-00001.parquet")
    nl = {}
    for name, xquad in (("dev-00000-of-00001.parquet", 0), ("test-00000-of-00001.parquet", 1)):
        t = _table(folder, name)
        for i, c, q, a in zip(t["id"], t["context"], t["question"], t["answers"]):
            nl[i] = (xquad, c, q, a)
    out, by_title, paragraphs = [], {}, {}
    for i, title, context, question, answers in zip(en["id"], en["title"], en["context"], en["question"],
                                                    en["answers"]):
        if title not in by_title:
            by_title[title] = {"title": title, "en": title.replace("_", " "), "nl": TITLES_NL[title], "paragraphs": []}
            out.append(by_title[title])
        if (title, context) not in paragraphs:
            paragraphs[title, context] = {"xquad": None, "shown": 1, "en": context, "nl": None, "questions": []}
            by_title[title]["paragraphs"].append(paragraphs[title, context])
        p = paragraphs[title, context]
        found = nl.get(i)
        if found is not None:
            xquad, nl_context, nl_question, nl_answers = found
            assert p["nl"] in (None, nl_context) and p["xquad"] in (None, xquad), (title, i)
            p["nl"], p["xquad"] = nl_context, xquad
        p["questions"].append({"id": i, "en": question, "en_answers": _answers(context, answers),
                               "nl": found[2] if found else None,
                               "nl_answers": _answers(found[1], found[3]) if found else []})
    for a in out:
        for p in a["paragraphs"]:
            assert p["nl"] is not None, a["title"]   # every paragraph has a Dutch question, and so its Dutch text
            if not p["xquad"]:
                p["questions"] = []
            p["shown"] = 0 if WORDS.search(p["en"]) or WORDS.search(p["nl"]) else 1
    return out


def main(folder: Path) -> None:
    found = articles(folder)
    data = json.dumps({"articles": found}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    OUT.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    ps = [p for a in found for p in a["paragraphs"]]
    qs = [q for p in ps for q in p["questions"]]
    print(f"{OUT}: {len(found)} articles, {len(ps)} paragraphs ({sum(p['xquad'] for p in ps)} of XQuAD, "
          f"{sum(1 - p['shown'] for p in ps)} not shown), {len(qs)} English and "
          f"{sum(q['nl'] is not None for q in qs)} Dutch questions, SHA-256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
