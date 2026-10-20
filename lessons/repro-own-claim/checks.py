# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-own-claim: verdict, the rule of the learner's own plan, which says from the runs of
two settings over the seeds of the plan whether they support the claim, saved as own/plan.py; and the review of the
plan itself, which the lesson's form fills in and which is saved as work/projects/own/plan.json once nothing in it
would be sent back. The course's plan for each paper of the list is here too, as the solution of that review."""
import importlib.util
import json
import re
from dataclasses import dataclass, field

from everything_ai import paths, projects
from everything_ai.check import Exercise, FunctionBox, Mistake, Result, _fill


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


O = _checks("repro-own-choose", "everything_ai_checks_repro_own_choose_for_claim")
LEVELS, HOURS, CARDS, EVENING = O.LEVELS, O.HOURS, O.CARDS, O.EVENING
papers, machine, saved_choice, plain, number = O.papers, O.machine, O.saved_choice, O.plain, O.number
PLAN = "plan.json"   # the learner's plan, in work/projects/own/, next to paper.json
NIGHT = 10           # hours: one night
MIN_SEEDS, MIN_MARGIN = 3, 2


def short(x: float, lang: str) -> str:
    """A number as the text writes it: 0.5, 1, -1.25; in Dutch with a decimal comma."""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    s = "0" if s in ("-0", "") else s
    return s.replace(".", ",") if lang == "nl" else s


# --- the play block: what counts --------------------------------------------------------------------------------------

# Made up, for the dropout paper: how many points lower the test error is with dropout than without, for each of three
# seeds, once when dropout works and once when it makes no difference and only the seeds differ.
WORKS = [0.9, 1.2, 1.5]
NO_EFFECT = [-0.3, 0.1, 0.4]


def _median(xs: list) -> float:
    s = sorted(xs)
    k = len(s) // 2
    return s[k] if len(s) % 2 else (s[k - 1] + s[k]) / 2


def counts_picture(at_least: float, labels: dict) -> str:
    """The two made-up outcomes as rows of dots, the gain of each seed, with the median of each row as a ring, and a
    dashed line between the rows at `at_least`, the smallest gain the plan counts as the claim: right of it, a row is
    labels["yes"], else labels["no"], said above the top row and below the bottom one, where the line does not reach.
    Ink: differences between runs, nothing a model learns. The panel can be dragged along its x-axis."""
    from everything_ai import chart
    p = chart.Panel((-1, 2.5), (-0.8, 1.75), title=labels["title"], xlabel=labels["x"], yticks=[])
    p.line([0, 0], [-0.25, 1.3], "ink-3", 1.0)
    p.line([at_least, at_least], [-0.25, 1.3], "ink", 2.0, dashed=True)
    for y, at, gains, name in ((1.0, 1.5, WORKS, labels["works"]), (0.0, -0.5, NO_EFFECT, labels["none"])):
        m = _median(gains)
        p.points([m], [y], "ink", r=8, hollow=True)
        p.points(gains, [y] * len(gains), "ink", r=4)
        p.text(-0.95, at, f"{name}: {labels['yes'] if m >= at_least else labels['no']}", "ink", "start")
    return chart.figure([p], alt=labels["alt"], width=360, height=250, drag=0)


def verdicts_at(at_least: float) -> tuple[bool, bool]:
    """Whether the plan counts each made-up outcome as the claim, at this `at_least`."""
    return _median(WORKS) >= at_least, _median(NO_EFFECT) >= at_least


# --- exercise 1: verdict ----------------------------------------------------------------------------------------------

_V_HEAD = ("import numpy as np\n\n\n"
           "def verdict(plan, results):\n"
           "    # \"too few runs\", \"supported\" or \"not supported\", by the rule of the plan;\n"
           "    # results[name] is the metric of each run of the setting `name`, seed 0 first\n"
           "    claim = plan[\"settings\"][0]   # the setting of the claim\n"
           "    other = plan[\"settings\"][1]   # the setting it is compared with\n"
           "    n = plan[\"seeds\"]\n")
_V_COUNT = ("    if len(results[claim]) < n or len(results[other]) < n:\n"
            "        return \"too few runs\"\n")
_V_GAINS = ("    gains = []   # for each seed: how far the setting of the claim is ahead, the way the claim says\n"
            "    for i in range(n):\n"
            "        if plan[\"direction\"] == \"higher\":\n"
            "            gains.append(results[claim][i] - results[other][i])\n"
            "        else:\n"
            "            gains.append(results[other][i] - results[claim][i])\n")
_V_RULE = ("    if np.median(gains) >= plan[\"at_least\"]:\n"
           "        return \"supported\"\n"
           "    return \"not supported\"\n")
_V = _V_HEAD + _V_COUNT + _V_GAINS + _V_RULE
_V_START = _V_HEAD + "    ...\n"

_V_ALL = _V.replace("for i in range(n):", "for i in range(len(results[claim])):")
_V_NO_COUNT = _V.replace(_V_COUNT, "")
_V_DIRECTION = _V.replace("gains.append(results[other][i] - results[claim][i])", "gains.append(results[claim][i] - results[other][i])")
_V_SWAPPED = (_V.replace("gains.append(results[claim][i] - results[other][i])", "GAIN_A")
              .replace("gains.append(results[other][i] - results[claim][i])", "gains.append(results[claim][i] - results[other][i])")
              .replace("GAIN_A", "gains.append(results[other][i] - results[claim][i])"))
_V_MEAN = _V.replace("np.median(gains)", "np.mean(gains)")
_V_STRICT = _V.replace("np.median(gains) >= plan", "np.median(gains) > plan")
_V_MIDDLE = _V.replace("np.median(gains)", "sorted(gains)[len(gains) // 2]")
_V_UNPAIRED = _V_HEAD + _V_COUNT + (
    "    if plan[\"direction\"] == \"higher\":\n"
    "        gain = np.median(results[claim][:n]) - np.median(results[other][:n])\n"
    "    else:\n"
    "        gain = np.median(results[other][:n]) - np.median(results[claim][:n])\n"
    "    if gain >= plan[\"at_least\"]:\n"
    "        return \"supported\"\n"
    "    return \"not supported\"\n")

_LOW = {"settings": ["with", "without"], "direction": "lower", "at_least": 1.0, "seeds": 3}
_WITHOUT = [12.0, 12.5, 11.0]
_VISIBLE = (_LOW, {"with": [10.0, 11.0, 10.5], "without": _WITHOUT})     # ahead by 2, 1.5 and 0.5: median 1.5
_HIDDEN = [
    (_LOW, {"with": [11.5, 12.0, 10.5], "without": _WITHOUT}),                      # ahead by 0.5 each time
    ({"settings": ["same start", "new start"], "direction": "higher", "at_least": 1.0, "seeds": 3},
     {"same start": [92.0, 93.0, 91.0], "new start": [89.0, 90.0, 90.0]}),          # higher is the claim: 3, 3 and 1
    (_LOW, {"with": [10.0, 11.0], "without": _WITHOUT}),                            # a seed short
    (_LOW, {"with": [10.0, 11.0, 10.5], "without": [12.0, 12.5]}),                  # the other a seed short
    (_LOW, {"with": [11.5, 12.0, 9.5, 10.0, 10.5], "without": _WITHOUT + [12.0, 12.5]}),   # 3 planned, 5 run
    (_LOW, {"with": [11.0, 11.5, 9.5], "without": _WITHOUT}),                       # median exactly at_least
    ({"settings": ["rank 4", "full"], "direction": "higher", "at_least": -1.0, "seeds": 3},
     {"rank 4": [80.0, 79.0, 81.5], "full": [81.0, 79.5, 83.0]}),                   # at most 1 behind: -1, -0.5, -1.5
    (_LOW, {"with": [15.0, 11.0, 9.0], "without": _WITHOUT}),                       # one seed far off: -3, 1.5, 2
    ({**_LOW, "at_least": 0.8}, {"with": [10.0, 13.0, 12.0], "without": [11.0, 15.0, 12.5]}),   # per seed 1, 2, 0.5
    ({**_LOW, "seeds": 4, "at_least": 1.8}, {"with": [10.0, 10.0, 10.0, 10.0], "without": [10.5, 11.0, 12.0, 13.0]}),
    ({**_LOW, "seeds": 4, "at_least": 1.2}, {"with": [10.0, 10.0, 10.0, 10.0], "without": [10.5, 11.0, 12.0, 13.0]}),   # 1.5: not the 1
]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

EXERCISES = {
    "verdict": FunctionBox(
        name="verdict",
        function="verdict",
        start=_V_START,
        solution=_V,
        visible=[_VISIBLE],
        hidden=_HIDDEN,
        python_values=True,
        save_as=("own", "plan"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-count", wrong=_V_NO_COUNT, error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, your function stops with an error: it asks for a run that is not there. Does it count the runs of both settings first, and give back `\"too few runs\"` when one has fewer than `n`? And does your loop go over the seeds 0 to `n - 1`, as `range(n)` does?",
                    nl="Met `{call}` stopt je functie met een fout: hij vraagt om een run die er niet is. Telt hij eerst de runs van beide settings, en geeft hij `\"too few runs\"` terug als er een minder dan `n` heeft? En gaat je loop over seed 0 tot en met `n - 1`, zoals `range(n)`?"),
            Mistake("all-runs", wrong=_V_ALL,
                    en="`{call}` gives `{got}`, and it should give `{want}`: the plan says 3 seeds, and you count all 5 runs. On the first 3 the claim is not supported; the 2 extra runs make it so. Runs you add after the plan do not count: that is cherry-picking. Loop over the first `n` only.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het plan zegt 3 seeds, en jij telt alle 5 runs. Op de eerste 3 wordt de claim niet gesteund; de 2 extra runs maken dat wel zo. Runs die je na het plan toevoegt, tellen niet: dat is cherry-picking. Loop alleen over de eerste `n`."),
            Mistake("direction", wrong=_V_DIRECTION,
                    en="`{call}` gives `{got}`, and it should give `{want}`: here the claim is that the metric is lower with its setting, and you count how much higher it is. When `plan[\"direction\"]` is `\"lower\"`, the gain of a seed is the other setting's metric minus that of the claim.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: hier is de claim dat de metric lager is met zijn setting, en jij telt hoeveel hoger hij is. Is `plan[\"direction\"]` gelijk aan `\"lower\"`, dan is de winst van een seed de metric van de andere setting min die van de claim."),
            Mistake("swapped", wrong=_V_SWAPPED,
                    en="`{call}` gives `{got}`, and it should give `{want}`: your two directions are swapped. With `\"higher\"`, the gain of a seed is the metric of the claim's setting minus the other's; with `\"lower\"`, the other way round.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je twee richtingen zijn omgedraaid. Met `\"higher\"` is de winst van een seed de metric van de setting van de claim min die van de andere; met `\"lower\"` andersom."),
            Mistake("mean", wrong=_V_MEAN,
                    en="`{call}` gives `{got}`, and it should give `{want}`: two seeds are well ahead and one is far behind, and with the mean that one seed decides. The plan takes the median, which one seed that goes wrong cannot move much: `np.median`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: twee seeds liggen ruim voor en één ligt ver achter, en met het gemiddelde beslist die ene seed. Het plan neemt de mediaan, die één seed die misgaat niet ver kan verschuiven: `np.median`."),
            Mistake("strict", wrong=_V_STRICT,
                    en="`{call}` gives `{got}`, and it should give `{want}`: the median is exactly `at_least`, and the plan says at least. That counts: `>=`, not `>`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de mediaan is precies `at_least`, en het plan zegt minstens. Dat telt: `>=`, niet `>`."),
            Mistake("unpaired", wrong=_V_UNPAIRED,
                    en="`{call}` gives `{got}`, and it should give `{want}`: you compare the median of each setting, and the plan compares per seed. Seed by seed, the claim's setting is ahead by 1, 2 and 0.5, a median of 1; the two medians differ by only 0.5. The same seed in both settings shares its luck: take the difference per seed first.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vergelijkt de mediaan van elke setting, en het plan vergelijkt per seed. Seed voor seed ligt de setting van de claim 1, 2 en 0,5 voor, een mediaan van 1; de twee medianen verschillen maar 0,5. Dezelfde seed in beide settings deelt zijn geluk: neem eerst het verschil per seed."),
            Mistake("middle", wrong=_V_MIDDLE,
                    en="`{call}` gives `{got}`, and it should give `{want}`: with 4 seeds there are two values in the middle, 1 and 2, and the median is halfway between them, 1.5. Your middle value is 2. `np.median` takes care of that.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: met 4 seeds staan er twee waarden in het midden, 1 en 2, en de mediaan ligt daar precies tussen, 1,5. Jouw middelste waarde is 2. `np.median` regelt dat."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with your code: count the runs, make the gain of each seed, and compare its median with `plan[\"at_least\"]`.",
                      "nl": "Vervang de `...` door je eigen code: tel de runs, maak de winst van elke seed, en vergelijk de mediaan met `plan[\"at_least\"]`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with a plan and the results, and the `def` line of your `verdict` has another number of names between its parentheses. Keep `(plan, results)`.",
                          "nl": "De check roept `{call}` aan, met een plan en de resultaten, en de `def`-regel van je `verdict` heeft een ander aantal namen tussen de haakjes. Houd `(plan, results)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you count the runs first, then make the gain of each seed the way the claim says, and compare the median of the gains with `plan[\"at_least\"]`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Tel je eerst de runs, maak je dan de winst van elke seed in de richting van de claim, en vergelijk je de mediaan van die winsten met `plan[\"at_least\"]`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your verdict work for every plan: a claim of higher and of lower, a seed short, more runs than planned, a median of exactly `at_least`, a negative `at_least`, and an even number of seeds?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je oordeel voor elk plan: een claim van hoger en van lager, een seed te weinig, meer runs dan gepland, een mediaan van precies `at_least`, een negatieve `at_least`, en een even aantal seeds?"},
            "correct": {"en": "Your verdict is right for all {n} plans: both directions, a seed short in either setting, two runs more than planned that do not count, the edge at exactly `at_least`, a claim of keeping up, one seed far off, the difference per seed, and four seeds. It is the rule of your plan now, as `{saved}`, written before your first run.",
                        "nl": "Je oordeel klopt voor alle {n} plannen: beide richtingen, een seed te weinig in een van beide settings, twee runs meer dan gepland die niet tellen, de rand bij precies `at_least`, een claim van even goed, één seed die ver afwijkt, het verschil per seed, en vier seeds. Het is nu de regel van je plan, als `{saved}`, opgeschreven vóór je eerste run."},
            "kept": {"en": "Your verdict is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je oordeel is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}


# --- the course's plans: one for each paper of the list ---------------------------------------------------------------

# What the course would plan for each paper, as a suggestion: the figure or table is checked against the paper (the
# arXiv version; the dropout paper's JMLR version, whose figures the text does not number); the smallest effect that
# counts is the course's choice, not measured yet, as the estimates are not.
PLANS = {
    "induction-heads": {
        # The card's claim is about timing, two moments close together either way, which one lower bound on a median
        # cannot say: the plan tests the part that it can, and the figure of the main result shows the timing.
        "claim": {"en": "Induction heads need two layers: a small transformer with two layers gets clearly better at in-context learning than one with a single layer, in which they cannot form.",
                  "nl": "Induction heads hebben twee layers nodig: een kleine transformer met twee layers wordt duidelijk beter in in-context learning dan een met één layer, waarin ze niet kunnen ontstaan."},
        "shown_in": {"en": "Argument 1: the phase change, in which the in-context learning score drops, in models with more than one layer",
                     "nl": "Argument 1: de faseovergang, waarin de in-context-learning-score daalt, bij modellen met meer dan één layer"},
        "metric": {"en": "the in-context learning score at the end: the loss at a late token of the context minus the loss at an early one",
                   "nl": "de in-context-learning-score aan het eind: de loss op een laat token van de context min die op een vroeg token"},
        "settings": {"en": ["two layers", "one layer"], "nl": ["twee layers", "één layer"]},
        "setup": {"en": "small attention-only transformers on TinyStories, with the same data, steps and learning rate; only the number of layers differs",
                  "nl": "kleine transformers met alleen attention op TinyStories, met dezelfde data, stappen en learning rate; alleen het aantal layers verschilt"},
        "direction": "lower", "at_least": 0.1},
    "emergence-mirage": {
        "shown_in": {"en": "Figure 3: the same models scored with an all-or-nothing metric and with a continuous one",
                     "nl": "Figuur 3: dezelfde modellen gescoord met een alles-of-niets-metric en met een continue"},
        "metric": {"en": "the largest rise between two neighbouring model sizes, as a share of the whole rise from the smallest model to the largest, from 0 to 1",
                   "nl": "de grootste stijging tussen twee naburige modelgroottes, als deel van de hele stijging van het kleinste model tot het grootste, van 0 tot 1"},
        "settings": {"en": ["a continuous metric", "exact match"], "nl": ["een continue metric", "exact match"]},
        "setup": {"en": "the same models of several sizes, trained with the code of llm-fit-scaling-law, one set for each seed, on the same task, up to a size where exact match rises above 0: below that, its rise is 0 and has no share; only the metric differs",
                  "nl": "dezelfde modellen van verschillende grootte, getraind met de code van llm-fit-scaling-law, één set per seed, op dezelfde taak, tot een grootte waarbij exact match boven 0 komt: daaronder is de stijging 0 en heeft ze geen deel; alleen de metric verschilt"},
        "direction": "lower", "at_least": 0.2},
    "reversal-curse": {
        "shown_in": {"en": "Figure 4: the log-probability of the right name and of a random one, asked the other way round",
                     "nl": "Figuur 4: de log-kans op de juiste naam en op een willekeurige, andersom gevraagd"},
        "metric": {"en": "the mean log-likelihood of the name, over the questions asked the other way round",
                   "nl": "de gemiddelde log-likelihood van de naam, over de vragen die andersom gesteld worden"},
        "settings": {"en": ["a random name", "the right name"], "nl": ["een willekeurige naam", "de juiste naam"]},
        "setup": {"en": "one model fine-tuned with LoRA on made-up facts written one way round, asked the other way round; only the name that is scored differs. Asked the way it learned them, the right name must be clearly more likely, or the fine-tuning did not take",
                  "nl": "één model, met LoRA gefinetuned op verzonnen feiten in één richting, andersom gevraagd; alleen de naam die gescoord wordt verschilt. In de richting waarin het ze leerde, moet de juiste naam duidelijk waarschijnlijker zijn, anders is het finetunen niet gelukt"},
        "direction": "higher", "at_least": -0.5},
    "lora-rank": {
        "shown_in": {"en": "Table 4: full fine-tuning and LoRA on GPT-3; Table 6: LoRA at ranks r from 1 to 64",
                     "nl": "Tabel 4: volledig finetunen en LoRA op GPT-3; Tabel 6: LoRA bij rangen r van 1 tot 64"},
        "metric": {"en": "the score on the task, in %", "nl": "de score op de taak, in %"},
        "settings": {"en": ["LoRA, rank 4", "full fine-tuning"], "nl": ["LoRA, rang 4", "volledig finetunen"]},
        "setup": {"en": "the same model, task, data and steps, with full fine-tuning clearly better than the model before; only which weights are trained differs",
                  "nl": "hetzelfde model, dezelfde taak, data en stappen, met volledig finetunen duidelijk beter dan het model ervoor; alleen welke weights getraind worden verschilt"},
        "direction": "higher", "at_least": -1.0},
    "random-labels": {
        "shown_in": {"en": "Figure 1a: the training loss with the true labels, random labels and random pixels; Table 1: the training accuracy",
                     "nl": "Figuur 1a: de training loss met de echte labels, willekeurige labels en willekeurige pixels; Tabel 1: de training-accuracy"},
        "metric": {"en": "the training accuracy at the end, in %", "nl": "de training-accuracy aan het eind, in %"},
        "settings": {"en": ["random labels", "the true labels"], "nl": ["willekeurige labels", "de echte labels"]},
        "setup": {"en": "the same network, pictures and number of steps, enough for the true labels to reach about 100%; only the labels differ",
                  "nl": "hetzelfde netwerk, dezelfde afbeeldingen en hetzelfde aantal stappen, genoeg om met de echte labels ongeveer 100% te halen; alleen de labels verschillen"},
        "direction": "higher", "at_least": -2.0},
    "lottery-ticket": {
        "shown_in": {"en": "Figure 4b: the test accuracy of winning tickets, and of the same tickets from new random starting weights",
                     "nl": "Figuur 4b: de test-accuracy van winning tickets, en van dezelfde tickets vanaf nieuwe willekeurige startwaarden"},
        "metric": {"en": "the test accuracy with 20% of the weights left, in %", "nl": "de test-accuracy met 20% van de weights over, in %"},
        "settings": {"en": ["its own starting weights", "new random starting weights"], "nl": ["zijn eigen startwaarden", "nieuwe willekeurige startwaarden"]},
        "setup": {"en": "the same network, data, steps and rounds of pruning; only the starting weights of the subnetwork differ",
                  "nl": "hetzelfde netwerk, dezelfde data, stappen en rondes pruning; alleen de startwaarden van het subnetwerk verschillen"},
        "direction": "higher", "at_least": 0.3},
    "dropout": {
        "shown_in": {"en": "the figure with the test error on MNIST of networks with and without dropout, as training goes on",
                     "nl": "de figuur met de testfout op MNIST van netwerken met en zonder dropout, terwijl de training loopt"},
        "metric": {"en": "the test error at the end, in %", "nl": "de testfout aan het eind, in %"},
        "settings": {"en": ["with dropout", "without dropout"], "nl": ["met dropout", "zonder dropout"]},
        "setup": {"en": "the same network on Fashion-MNIST, with the same steps and learning rate; only dropout differs",
                  "nl": "hetzelfde netwerk op Fashion-MNIST, met dezelfde stappen en learning rate; alleen dropout verschilt"},
        "direction": "lower", "at_least": 0.5},
    "resnet-degradation": {
        "shown_in": {"en": "Figure 1: the training error of plain networks of 20 and 56 layers on CIFAR-10",
                     "nl": "Figuur 1: de trainingsfout van gewone netwerken van 20 en 56 layers op CIFAR-10"},
        "metric": {"en": "the training error at the end, in %", "nl": "de trainingsfout aan het eind, in %"},
        "settings": {"en": ["a deep plain network", "a shallow plain network"], "nl": ["een diep gewoon netwerk", "een ondiep gewoon netwerk"]},
        "setup": {"en": "plain networks of the same kind of layers on Fashion-MNIST, with the same steps; only the number of layers differs, such as 32 against 20",
                  "nl": "gewone netwerken van dezelfde soort layers op Fashion-MNIST, met dezelfde stappen; alleen het aantal layers verschilt, zoals 32 tegen 20"},
        "direction": "higher", "at_least": 1.0},
}


def course_plan(paper_id: str, lang: str) -> dict:
    """The course's plan for a paper of the list, in the shape of the review: its card's claim, three seeds, the
    estimate split over them, a margin of 2, the least computer it needs, and an evening, or a night when it does not
    fit one."""
    paper = next(p for p in papers() if p["id"] == paper_id)
    s = PLANS[paper_id]
    per_seed = int(10 * paper["hours"] / MIN_SEEDS) / 10   # rounded down, to tenths
    cost = MIN_SEEDS * per_seed * MIN_MARGIN
    return {"paper": paper_id, "needs": paper["needs"], "claim": s.get("claim", CARDS[paper_id]["claim"])[lang],
            "shown_in": s["shown_in"][lang], "metric": s["metric"][lang], "settings": list(s["settings"][lang]),
            "setup": s["setup"][lang], "direction": s["direction"], "at_least": s["at_least"], "seeds": MIN_SEEDS,
            "hours_per_seed": per_seed, "margin": MIN_MARGIN, "computer": paper["needs"],
            "hours": EVENING if cost <= EVENING else NIGHT}


# --- exercise 2: the plan, and its review ------------------------------------------------------------------------------

def claim_of(choice: dict | None, lang: str) -> str:
    """The claim the plan starts with: the card's, in the lesson's language, for a paper of the list; the learner's
    own for their own paper; nothing when there is no choice yet."""
    if not choice:
        return ""
    return CARDS[choice["id"]]["claim"][lang] if choice.get("id") in CARDS else str(choice.get("claim", ""))


def _text(x) -> str:
    return x.strip() if isinstance(x, str) else ""


def _num(x) -> float | None:
    try:
        return None if x is None or isinstance(x, bool) else float(x)
    except (TypeError, ValueError):
        return None


def _settings(plan: dict) -> tuple[str, str]:
    s = list(plan.get("settings") or []) + ["", ""]
    return _text(s[0]), _text(s[1])


def cost(plan: dict) -> float:
    """What the plan costs, in hours: every seed, both settings, times the margin."""
    return (_num(plan.get("seeds")) or 0) * (_num(plan.get("hours_per_seed")) or 0) * (_num(plan.get("margin")) or 0)


def filled(plan: dict) -> bool:
    """Something of the plan is the learner's: a word or a number in one of the fields the form starts empty."""
    return any(_text(plan.get(k)) for k in ("shown_in", "metric", "setup")) or any(_settings(plan)) \
        or bool(_num(plan.get("at_least"))) or bool(_num(plan.get("hours_per_seed")))


def problem(plan: dict) -> tuple[str, dict] | None:
    """The first thing a reviewer would send back, as the key of its message and the values the message names, or
    None when there is nothing: in the order the form asks for them."""
    a, b = _settings(plan)
    seeds, at_least = _num(plan.get("seeds")) or 0, _num(plan.get("at_least"))
    per_seed, margin, hours = _num(plan.get("hours_per_seed")) or 0, _num(plan.get("margin")) or 0, _num(plan.get("hours")) or 0
    needs, computer = plan.get("needs"), plan.get("computer")
    if plan.get("paper") is None or needs not in LEVELS:
        return "no-paper", {}
    if not _text(plan.get("claim")):
        return "claim", {}
    if not _text(plan.get("shown_in")):
        return "shown-in", {}
    if not (a and b):
        return "settings", {}
    if a.lower() == b.lower():
        return "same-settings", {"a": a}
    if not _text(plan.get("setup")):
        return "setup", {}
    if not _text(plan.get("metric")):
        return "metric", {}
    if seeds < MIN_SEEDS or seeds != int(seeds):
        none = _num(plan.get("seeds")) is None   # an emptied number field
        return "seeds", {"n": {lang: ({"en": "no", "nl": "geen"}[lang] if none else short(seeds, lang)) for lang in ("en", "nl")}}
    if plan.get("direction") not in ("higher", "lower"):
        return "direction", {}
    if not at_least:
        return "zero", {}
    if per_seed <= 0:
        return "hours", {}
    if margin < MIN_MARGIN:
        return "margin", {}
    if computer not in LEVELS or LEVELS.index(needs) > LEVELS.index(computer):
        return "level", {}
    if round(cost(plan), 6) > hours:   # 3 * 0.4 * 2.5 is 3.0000000000000004
        return "cost", {"cost": {lang: short(cost(plan), lang) for lang in ("en", "nl")},
                        "hours": {lang: short(hours, lang) for lang in ("en", "nl")}}
    return None


def rule_text(plan: dict, lang: str) -> str:
    """The rule of the plan as one sentence: when the runs support the claim, when not, and when there is no verdict."""
    a, b = (plain(s) for s in _settings(plan))
    d, n, metric = _num(plan.get("at_least")) or 0, int(_num(plan.get("seeds")) or 0), plain(_text(plan.get("metric")))
    up = plan.get("direction") == "higher"
    words = {"en": ("higher", "lower"), "nl": ("hoger", "lager")}[lang]
    if lang == "en":
        how = f"at least {short(d, lang)} {words[0] if up else words[1]}" if d >= 0 else f"no more than {short(-d, lang)} {words[1] if up else words[0]}"
        return (f"Supported when, compared seed by seed over seeds 0 to {n - 1}, the metric is in the median {how} for *{a}* than for *{b}* (the metric: {metric}); "
                f"not supported when it is not; and no verdict until both settings have {n} runs.")
    how = f"minstens {short(d, lang)} {words[0] if up else words[1]}" if d >= 0 else f"hooguit {short(-d, lang)} {words[1] if up else words[0]}"
    return (f"Gesteund als, seed voor seed vergeleken over seed 0 tot en met {n - 1}, de metric in de mediaan {how} is voor *{a}* dan voor *{b}* (de metric: {metric}); "
            f"niet gesteund als dat niet zo is; en geen oordeel zolang beide settings nog geen {n} runs hebben.")


def plan_path():
    return projects.folder() / "own" / PLAN


def saved_plan() -> dict | None:
    try:
        p = json.loads(plan_path().read_text(encoding="utf-8"))
        return p if isinstance(p, dict) and p.get("paper") else None
    except (OSError, ValueError):
        return None


def save_plan(plan: dict) -> str:
    """Saves the plan as work/projects/own/plan.json, with numbers as numbers and the date it was written: the date of
    the plan as it was saved before when nothing in it changed, so that later lessons can show it came before the runs.
    Gives back where, as the lesson shows it."""
    import datetime
    keep = {**plan, "settings": list(_settings(plan))}
    for k in ("at_least", "seeds", "hours_per_seed", "margin", "hours"):
        keep[k] = _num(plan.get(k))
    keep["seeds"] = int(keep["seeds"])
    keep.pop("written", None)
    old = saved_plan() or {}
    keep["written"] = old.get("written") if old and {k: v for k, v in old.items() if k != "written"} == keep else None
    keep["written"] = keep["written"] or datetime.date.today().isoformat()
    path = plan_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return projects.shown(path)


@dataclass
class PlanReview(Exercise):
    """The learner's plan, a dict the lesson's form fills in: the check sends back the first thing a reviewer would,
    and saves the plan once there is nothing. `start` is a plan with only what the form fills in itself."""
    start: dict = field(default_factory=dict)
    solution: dict = field(default_factory=dict)

    def check(self, plan) -> Result:
        if not isinstance(plan, dict):
            return Result("error", self.message("missing"))
        if not filled(plan):
            return Result("empty", self.message("empty"))
        found = problem(plan)
        if found is not None:
            key, values = found
            m = next(m for m in self.mistakes if m.key == key)
            return Result(m.state, _fill(m.text, **values), mistake=key)
        saved = save_plan(plan)
        return Result("correct", self.message("correct", saved=saved, rule={lang: rule_text(plan, lang) for lang in ("en", "nl")},
                                              cost={lang: short(cost(plan), lang) for lang in ("en", "nl")},
                                              hours={lang: short(_num(plan.get("hours")), lang) for lang in ("en", "nl")}))


_P = course_plan("dropout", "en")
_P_START = {**_P, "shown_in": "", "metric": "", "settings": ["", ""], "setup": "", "direction": None, "at_least": 0,
            "hours_per_seed": 0, "computer": "any", "hours": EVENING}
_P_WRONG = {
    "no-paper": {**_P, "paper": None, "needs": None},
    "claim": {**_P, "claim": " "},
    "shown-in": {**_P, "shown_in": ""},
    "metric": {**_P, "metric": ""},
    "settings": {**_P, "settings": ["with dropout", ""]},
    "same-settings": {**_P, "settings": ["with dropout", "With dropout "]},
    "setup": {**_P, "setup": ""},
    "direction": {**_P, "direction": None},
    "seeds": {**_P, "seeds": 1},
    "zero": {**_P, "at_least": 0},
    "level": {**course_plan("lora-rank", "en"), "computer": "gpu8"},
    "hours": {**_P, "hours_per_seed": 0},
    "margin": {**_P, "margin": 1},
    "cost": {**_P, "hours_per_seed": 1.0},
}

EXERCISES["plan"] = PlanReview(
    name="plan",
    start=_P_START,
    solution=_P,
    mistakes=[
        Mistake("no-paper", wrong=_P_WRONG["no-paper"],
                en="Your plan is not for a paper yet: choose it in [Your reproduction 1](/course/open/repro-own-choose?lang=en) and save your choice, then open this lesson again.",
                nl="Je plan hoort nog bij geen paper: kies die in [Je reproductie 1](/course/open/repro-own-choose?lang=nl) en bewaar je keuze, en open deze les dan opnieuw."),
        Mistake("claim", wrong=_P_WRONG["claim"],
                en="Your plan has no claim. Write the claim you test in one sentence: it starts as the claim of your paper.",
                nl="Je plan heeft geen claim. Schrijf de claim die je test in één zin: hij begint als de claim van je paper."),
        Mistake("shown-in", wrong=_P_WRONG["shown-in"],
                en="Where does the paper show the claim? Name the figure or table, so you know what your result should look like, and so does your reader.",
                nl="Waar laat de paper de claim zien? Noem de figuur of tabel, zodat jij weet hoe je resultaat eruit moet zien, en je lezer ook."),
        Mistake("metric", wrong=_P_WRONG["metric"],
                en="What do you measure? Name the metric: one number per run, such as the test error at the end, in %.",
                nl="Wat meet je? Noem de metric: één getal per run, zoals de testfout aan het eind, in %."),
        Mistake("settings", wrong=_P_WRONG["settings"],
                en="A claim compares: name both settings, the one of the claim and the one it is compared with, such as with and without dropout.",
                nl="Een claim vergelijkt: noem beide settings, die van de claim en die waarmee je hem vergelijkt, zoals met en zonder dropout."),
        Mistake("same-settings", wrong=_P_WRONG["same-settings"],
                en="Both settings are called *{a}*. The setting of the claim needs another to be compared with: what is different in the other one?",
                nl="Beide settings heten *{a}*. De setting van de claim heeft een andere nodig om mee te vergelijken: wat is er anders in de andere?"),
        Mistake("setup", wrong=_P_WRONG["setup"],
                en="What stays the same in both settings? Name the data, the network and the steps, so that only one thing differs: then a difference is caused by that one thing.",
                nl="Wat blijft hetzelfde in beide settings? Noem de data, het netwerk en de stappen, zodat er maar één ding verschilt: dan komt een verschil door dat ene ding."),
        Mistake("direction", wrong=_P_WRONG["direction"],
                en="Which way does the claim go? Choose whether the metric is higher or lower with the setting of the claim.",
                nl="Welke kant gaat de claim op? Kies of de metric hoger of lager is met de setting van de claim."),
        Mistake("seeds", wrong=_P_WRONG["seeds"],
                en="Your plan has {n} seeds. With fewer than 3, one seed that goes wrong decides, and a median over one or two runs is no better than the runs themselves: plan 3 seeds or more, a whole number.",
                nl="Je plan heeft {n} seeds. Met minder dan 3 beslist één seed die misgaat, en een mediaan over één of twee runs is niet beter dan de runs zelf: plan 3 seeds of meer, een heel getal."),
        Mistake("zero", wrong=_P_WRONG["zero"],
                en="Your plan counts every difference, however small: then a difference that is only noise between seeds supports the claim as well. Say how big it has to be, from the claim and from how much two seeds differ.",
                nl="Je plan telt elk verschil, hoe klein ook: dan steunt een verschil dat alleen ruis tussen seeds is de claim net zo goed. Zeg hoe groot het moet zijn, uit de claim en uit hoeveel twee seeds verschillen."),
        Mistake("level", wrong=_P_WRONG["level"],
                en="Your paper needs more than the computer of your plan has, and more hours do not help: choose another paper in [Your reproduction 1](/course/open/repro-own-choose?lang=en), or plan on a computer that has what it needs.",
                nl="Je paper heeft meer nodig dan de computer van je plan heeft, en meer uren helpen niet: kies een andere paper in [Je reproductie 1](/course/open/repro-own-choose?lang=nl), of plan op een computer die heeft wat hij nodig heeft."),
        Mistake("hours", wrong=_P_WRONG["hours"],
                en="How long does one seed take, both settings together? Time one short run, for a paper of the list in the lesson it builds on, and multiply by the steps you plan.",
                nl="Hoe lang duurt één seed, beide settings samen? Meet de tijd van één korte run, voor een paper van de lijst in de les waarop hij bouwt, en vermenigvuldig met de stappen die je plant."),
        Mistake("margin", wrong=_P_WRONG["margin"],
                en="With a margin below 2 you count on your estimate being right, and a first plan rarely is: a run that has to be done again, a bug, a slower step. Plan with a margin of 2 or 3.",
                nl="Met een marge onder 2 reken je erop dat je schatting klopt, en dat doet een eerste plan zelden: een run die over moet, een bug, een tragere stap. Plan met een marge van 2 of 3."),
        Mistake("cost", wrong=_P_WRONG["cost"],
                en="Your plan costs {cost} hours with its margin, and you have {hours}. Plan more time, such as a night instead of an evening, or shorter runs. Fewer seeds than 3 is no way out.",
                nl="Je plan kost {cost} uur met de marge, en je hebt er {hours}. Plan meer tijd, zoals een nacht in plaats van een avond, of kortere runs. Minder dan 3 seeds is geen uitweg."),
    ],
    texts={
        "missing": {"en": "I cannot see your plan. Run the cell with the form again.",
                    "nl": "Ik zie je plan niet. Voer de cel met het formulier opnieuw uit."},
        "empty": {"en": "Fill in your plan above: where the paper shows the claim, the two settings, what stays the same, what you measure, which way and by how much, and what one seed costs.",
                  "nl": "Vul je plan hierboven in: waar de paper de claim laat zien, de twee settings, wat hetzelfde blijft, wat je meet, welke kant op en hoeveel, en wat één seed kost."},
        "correct": {"en": "A reviewer would send nothing back: your plan is saved as `{saved}`. Its rule: {rule} It costs {cost} hours with its margin, of the {hours} you have.",
                    "nl": "Een reviewer zou niets terugsturen: je plan staat nu in `{saved}`. De regel: {rule} Het kost {cost} uur met de marge, van de {hours} die je hebt."},
    },
)

# The lesson is done when these are correct. "why-number" is the open question.
REQUIRED = ["verdict", "plan", "why-number"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("verdict", _V_START, "empty", None),
    ("verdict", _V, "correct", None),
    ("verdict", _V_HEAD + _V_COUNT + "    gains = []\n    for i in range(n):\n        gain = results[claim][i] - results[other][i]\n"
     "        if plan[\"direction\"] == \"lower\":\n            gain = -gain\n        gains.append(gain)\n"
     "    return \"supported\" if np.median(gains) >= plan[\"at_least\"] else \"not supported\"\n", "correct", None),
    ("verdict", _V.replace("np.median(gains)", "sorted(gains)[(len(gains) - 1) // 2]"), "almost", None),   # the lower middle value
    ("verdict", _V.replace(" or len(results[other]) < n", ""), "error", "no-count"),                        # counts one setting only
    ("verdict", _V.replace("range(n)", "range(1, n + 1)"), "error", "no-count"),                            # seeds from 1
    ("verdict", _V_HEAD + _V_COUNT + "    a = np.array(results[claim][:n])\n    b = np.array(results[other][:n])\n"
     "    gains = a - b if plan[\"direction\"] == \"higher\" else b - a\n"
     "    if np.median(gains) < plan[\"at_least\"]:\n        return \"not supported\"\n    return \"supported\"\n", "correct", None),
    ("plan", _P_START, "empty", None),
    ("plan", _P, "correct", None),
    ("plan", course_plan("emergence-mirage", "nl"), "correct", None),
    ("plan", {**_P, "seeds": 3.5}, "wrong", "seeds"),
    ("plan", {**_P, "at_least": None}, "wrong", "zero"),
    ("plan", {**_P, "settings": None}, "wrong", "settings"),
    ("plan", {**_P, "seeds": "3", "hours_per_seed": "0.3", "margin": "2", "hours": "4"}, "correct", None),
    ("plan", {**_P, "hours_per_seed": 0.4, "margin": 2.5, "hours": 3}, "correct", None),   # costs exactly the 3 hours
] + [("verdict", code, "error" if key == "no-count" else "wrong", key)
     for key, code in (("no-count", _V_NO_COUNT), ("all-runs", _V_ALL), ("direction", _V_DIRECTION), ("swapped", _V_SWAPPED),
                       ("mean", _V_MEAN), ("strict", _V_STRICT), ("unpaired", _V_UNPAIRED), ("middle", _V_MIDDLE))] \
  + [("plan", plan, "wrong", key) for key, plan in _P_WRONG.items()]


# --- the demo: your rule on made-up outcomes ---------------------------------------------------------------------------

def outcomes(plan: dict) -> dict:
    """Three made-up outcomes for the plan, as results by setting: one where the claim's setting is ahead by about
    twice what counts, one where it falls short by as much, and the first of them stopped a seed early. The other
    setting is at 0 on every seed: only the gains matter to the rule."""
    a, b = _settings(plan)
    n, d = int(_num(plan.get("seeds")) or 0), _num(plan.get("at_least")) or 0
    step = max(abs(d), 0.1)
    jitter = [0.2 * step * ((i % 3) - 1) for i in range(n)]
    sign = 1 if plan.get("direction") == "higher" else -1
    holds = [round(sign * (d + step + j), 4) for j in jitter]
    fails = [round(sign * (d - step + j), 4) for j in jitter]
    return {"holds": {a: holds, b: [0.0] * n}, "fails": {a: fails, b: [0.0] * n}, "early": {a: holds[:-1], b: [0.0] * (n - 1)}}


def demo(plan: dict) -> tuple[dict, str] | None:
    """The verdict of the learner's own plan.py on the three outcomes, and whose verdict it was: "yours", or "course"
    when the file is not there yet or stops with an error. None when the plan cannot be judged yet."""
    import importlib
    if problem(plan) is not None:
        return None
    projects.on_path()
    out = outcomes(plan)
    if (projects.folder() / "own" / "plan.py").exists():
        projects.forget()
        try:
            mine = importlib.import_module("own.plan")
            return {k: mine.verdict(plan, r) for k, r in out.items()}, "yours"
        except Exception:
            pass
    from everything_ai.check import run_code
    course = run_code(_V).names["verdict"]
    return {k: course(plan, r) for k, r in out.items()}, "course"


def outcomes_table(plan: dict, verdicts: dict, labels: dict, lang: str) -> str:
    """The three made-up outcomes of `outcomes` as a table: the gain of each seed, and the verdict on it."""
    import html
    a, b = _settings(plan)
    up = plan.get("direction") == "higher"
    pad = "padding:3px 12px 3px 0;vertical-align:top"
    out = [f"<table class='ea-cards' style='border-collapse:collapse'><thead><tr><th style='{pad};text-align:left'>{labels['outcome']}</th>"
           f"<th style='{pad};text-align:left'>{labels['gains']}</th><th style='{pad};text-align:left'>{labels['verdict']}</th></tr></thead><tbody>"]
    for key, r in outcomes(plan).items():
        gains = [x - y if up else y - x for x, y in zip(r[a], r[b])]
        shown = ", ".join(short(g, lang).replace("-", "−") for g in gains)
        out.append(f"<tr><td style='{pad}'>{labels[key]}</td><td style='{pad};white-space:nowrap'>{html.escape(shown)}</td>"
                   f"<td style='{pad};white-space:nowrap'><code>{html.escape(str(verdicts[key]))}</code></td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


def course_plan_text(choice: dict | None, lang: str) -> str:
    """The course's plan for the learner's paper, or for Dropout as an example when it is their own or none, in
    Markdown: the solution of exercise 2."""
    pid = choice["id"] if choice and choice.get("id") in PLANS else "dropout"
    p, paper = course_plan(pid, lang), next(x for x in papers() if x["id"] == pid)
    a, b = p["settings"]
    c, h = short(cost(p), lang), short(p["hours"], lang)
    if lang == "nl":
        head = (f"Het plan van de course voor {plain(paper['title'])}:" if pid == (choice or {}).get("id")
                else "Je paper is een eigen paper, of nog niet gekozen; hier is het plan van de course voor Dropout, als voorbeeld:")
        return (f"{head}\n\n- **Claim:** {plain(p['claim'])}\n- **Te zien in:** {plain(p['shown_in'])}\n"
                f"- **Settings:** *{plain(a)}* tegen *{plain(b)}*; hetzelfde blijft: {plain(p['setup'])}\n- **Metric:** {plain(p['metric'])}\n"
                f"- **Regel:** {rule_text(p, lang)}\n- **Kosten:** {p['seeds']} seeds van {short(p['hours_per_seed'], lang)} uur, marge {p['margin']}: {c} uur, van de {h} die je hebt.\n\n"
                "Het getal achter *minstens* is een keuze van de course, nog niet gemeten: leg het naast hoeveel je eigen seeds verschillen.")
    head = (f"The course's plan for {plain(paper['title'])}:" if pid == (choice or {}).get("id")
            else "Your paper is one of your own, or not chosen yet; here is the course's plan for Dropout, as an example:")
    return (f"{head}\n\n- **Claim:** {plain(p['claim'])}\n- **Shown in:** {plain(p['shown_in'])}\n"
            f"- **Settings:** *{plain(a)}* against *{plain(b)}*; the same in both: {plain(p['setup'])}\n- **Metric:** {plain(p['metric'])}\n"
            f"- **Rule:** {rule_text(p, lang)}\n- **Cost:** {p['seeds']} seeds of {short(p['hours_per_seed'], lang)} hours, margin {p['margin']}: {c} hours, of the {h} you have.\n\n"
            "The number of *at least* is the course's choice, not measured yet: hold it against how much your own seeds differ.")
