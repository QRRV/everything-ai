# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/wordnet-en.tsv.gz: the synsets of Open English WordNet 2025 (CC-BY-4.0 and the WordNet License), one per
row, with their words, their definition and the relations the lessons about word meanings use. Every synset gets a name
as NLTK gives one: its first word, its part of speech and the sense number of that word, such as dog.n.01. Where the
Collaborative Interlingual Index (Bond and others, 2016; CC-BY-4.0) links a synset to Princeton WordNet 3.0, the row
also has that synset's offset (such as 02084071-n, with -a also for an adjective in a cluster), the key of
data/wordnet-nl.tsv, so that the Dutch words of Open Dutch WordNet join the English synsets.

Left out: synsets whose definition says that the word is an ethnic slur, or an offensive, derogatory, disparaging,
obscene or vulgar term (OFFENSIVE below), and every relation to them. No filter of this kind is complete. Download the
two files at the fixed versions below and run, from the course folder:

    uv run python data/tools/make_wordnet_en.py english-wordnet-2025.xml.gz ili-map-pwn30.tab

Columns, after a header row: name; pwn30 (empty where there is none); pos (n, v, a, s for an adjective in a cluster,
r); words (the synset's words in order, each with its sense number for that part of speech after a colon, such as
dog:1, with _ for a space); definition; hypernym, part, member, substance (the names of the synsets that relation leads
to, separated by spaces: the hypernyms, past a synset that is left out to its hypernyms; the parts, members and
substances the synset has); antonym (word>name pairs,
such as hot:1>cold.a.01, separated by spaces: the antonyms of the synset's words).
"""
import gzip
import hashlib
import re
import sys
import xml.etree.ElementTree as ET

from everything_ai import paths

SOURCE = "https://github.com/globalwordnet/english-wordnet/releases/download/2025-edition/english-wordnet-2025.xml.gz"
SHA256 = "9ca6d1dcb75f822fdd66617f7d9da48142ace38dd544d6ad5e2feca1674ad3fe"
ILI_SOURCE = "https://raw.githubusercontent.com/globalwordnet/cili/a895d7ecb18019dda3443f98901e59d81ce8722b/ili-map-pwn30.tab"
ILI_SHA256 = "13b6741395a8bdf20e5a19686ffa49167653696d4f523db17a79dce3211a6627"

OFFENSIVE = re.compile(
    r"ethnic slur|\((?:sometimes |often )?offensive|offensive (?:british )?slang|\(slang\) offensive|terms? of abuse"
    r"|\b(?:offensive|derogatory|disparaging|insulting)(?: and insulting)? (?:term|terms|name|names|word|words)\b"
    r"|\b(?:obscene|vulgar) (?:term|terms|word|words|slang|expression)\b|\bslang for (?:sexual|masturbation)",
    re.IGNORECASE)
# Words left out wherever they occur: slurs, and crude words for sex, the body and its waste. A synset that keeps none of
# its words is left out too. Words that are fine in other senses (such as cock for a rooster) stay, without the synsets
# OFFENSIVE leaves out.
DROP = {
    "Mongoloid", "Negro", "Negroid", "arse", "arsehole", "asshole", "bastard", "beaner", "bitch", "bitchy", "bugger",
    "bullshit", "coon", "crap", "crappy", "cum", "cunt", "fag", "fagot", "faggot", "fuck", "fucker", "fucking", "gook",
    "gringo", "harlot", "horseshit", "hussy", "midget", "mongoloid", "negro", "papist", "piss", "pissed", "retard",
    "shit", "shitty", "shylock", "slut", "sluttish", "spaz", "spic", "strumpet", "trollop", "twat", "wank", "wanker",
    "whore",
}
RELATIONS = {"hypernym": "hypernym", "mero_part": "part", "mero_member": "member", "mero_substance": "substance"}
COLUMNS = ["name", "pwn30", "pos", "words", "definition", "hypernym", "part", "member", "substance", "antonym"]


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(xml_gz: str, ili_tab: str) -> None:
    assert _sha(xml_gz) == SHA256, xml_gz
    assert _sha(ili_tab) == ILI_SHA256, ili_tab
    pwn30 = {}
    with open(ili_tab, encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                ili, offset = line.split("\t")[:2]
                offset = offset.strip()
                pwn30[ili] = offset[:-2] + "-a" if offset.endswith("-s") else offset   # as data/wordnet-nl.tsv writes it
    entry_form, order, synsets, sense_of, antonyms = {}, {}, {}, {}, []
    for _, el in ET.iterparse(gzip.open(xml_gz), events=("end",)):
        if el.tag == "LexicalEntry":
            lemma = el.find("Lemma")
            form, pos = lemma.get("writtenForm"), lemma.get("partOfSpeech")
            entry_form[el.get("id")] = form
            for sense in el.findall("Sense"):
                order.setdefault((form, pos), []).append(sense.get("synset"))   # the senses of a word, most common first
                sense_of[sense.get("id")] = (form, pos, sense.get("synset"))
                for r in sense.findall("SenseRelation"):
                    if r.get("relType") == "antonym":
                        antonyms.append((sense.get("id"), r.get("target")))
            el.clear()
        elif el.tag == "Synset":
            rels = {}
            for r in el.findall("SynsetRelation"):
                if r.get("relType") in RELATIONS:
                    rels.setdefault(RELATIONS[r.get("relType")], []).append(r.get("target"))
            synsets[el.get("id")] = {"pos": el.get("partOfSpeech"), "ili": el.get("ili") or "",
                                     "members": [entry_form[m] for m in el.get("members", "").split()],
                                     "definition": " ".join((el.findtext("Definition") or "").split()), "rels": rels}
            el.clear()
    assert DROP <= {form for form, _ in order}, DROP - {form for form, _ in order}   # every word of DROP is in the file
    offensive = {k for k, s in synsets.items() if OFFENSIVE.search(s["definition"])}
    kept = {k: [m for m in s["members"] if m not in DROP] for k, s in synsets.items() if k not in offensive}
    kept = {k: members for k, members in kept.items() if members}
    # sense numbers among the senses that are kept, per word and part of speech
    number = {}
    for (form, pos), synset_ids in order.items():
        if form not in DROP:
            for i, k in enumerate([k for k in synset_ids if k in kept], 1):
                number[(form, k)] = i
    names = {k: f"{members[0].replace(' ', '_')}.{synsets[k]['pos']}.{number[(members[0], k)]:02d}" for k, members in kept.items()}
    assert len(set(names.values())) == len(names), "names are unique"
    ant = {}
    for source, target in antonyms:
        form, _, k = sense_of[source]
        other_form, _, other = sense_of[target]
        if k in kept and other in kept and form not in DROP and other_form not in DROP:
            ant.setdefault(k, []).append(f"{form.replace(' ', '_')}:{number[(form, k)]}>{names[other]}")
    def up(k: str) -> list:
        """The hypernyms of k that are kept: past a synset that is left out, to its own hypernyms."""
        out = []
        for t in synsets[k]["rels"].get("hypernym", []):
            for u in ([t] if t in kept else up(t)):
                if u not in out:
                    out.append(u)
        return out

    rows = []
    for k, members in kept.items():
        s = synsets[k]
        words = " ".join(f"{m.replace(' ', '_')}:{number[(m, k)]}" for m in members)
        rel = [" ".join(names[t] for t in up(k))]
        rel += [" ".join(names[t] for t in s["rels"].get(r, []) if t in kept) for r in ("part", "member", "substance")]
        rows.append([names[k], pwn30.get(s["ili"], ""), s["pos"], words, s["definition"], *rel, " ".join(ant.get(k, []))])
    for row in rows:
        assert not any("\t" in x or "\n" in x for x in row), row[0]
    text = "\t".join(COLUMNS) + "\n" + "".join("\t".join(r) + "\n" for r in rows)
    out = paths.ROOT / "data" / "wordnet-en.tsv.gz"
    with open(out, "wb") as f:
        f.write(gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0))
    print(len(rows), "synsets,", len(synsets) - len(rows), "left out,", sum(1 for r in rows if r[1]),
          "with a Princeton WordNet 3.0 offset,", out.stat().st_size, "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:])
