# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-project-bandit: the start of the learner's project agent. BanditAgent, an agent as a class
with act and learn; run, the loop in which an agent and a bandit take turns; average, the mean over the runs at every
pull; DriftingBandit, the course's testbed in agent/testbed.py; the demo, in which three versions of the learner's
agent play the same 200 drifting bandits for 10,000 pulls; and the pictures.

The testbed follows Sutton and Barto (Reinforcement Learning: An Introduction, 2nd edition, 2018, section 2.3 and
exercise 2.5): ten machines whose means start around 0 with standard deviation 1, here drifting at every pull, and
results averaged over many bandits. The code, the runs and the pictures are the course's own."""
import ast
import copy
import html
import math
import numbers
from pathlib import Path

import numpy as np

from everything_ai import chart, live, projects, yours
from everything_ai.check import ClassBox, FunctionBox, Mistake, ScenarioBox, Timeout, kept, run_code, with_time_limit

LESSON = "rl-project-bandit"
# Exercise 2 lets your agent of exercise 1 play: while agent/bandit.py is missing or broken, the course's stands in.
projects.builds_on(("agent", "bandit", LESSON, "agent"))
projects.provide(LESSON)   # agent/testbed.py, which the lesson gives, also when only the checks are loaded

K = 10             # machines of a bandit
DRIFT = 0.01       # the standard deviation of the step every mean takes at every pull
ETA = 0.1          # the constant learning rate of the three agents
EPSILON = 0.1      # the epsilon of the epsilon-greedy agent
START = 5.0        # the optimistic start
AGENTS = ("greedy", "epsilon", "optimistic")
SETTINGS = {"greedy": (0.0, ETA, 0.0), "epsilon": (EPSILON, ETA, 0.0), "optimistic": (0.0, ETA, START)}   # epsilon, eta, start

DEMO_BANDITS = 200   # bandits of the demo; bandit b gets np.random.default_rng(BANDIT_SEED + b)
PULLS = 10_000       # pulls of every run of the demo
BLOCK = 100          # pulls of each call of run in the demo: after each block the chart grows
BANDIT_SEED = 0
AGENT_SEED = 1000    # the agent on bandit b gets np.random.default_rng(AGENT_SEED + b)
WATCHED = 106        # the bandit whose machines the picture beside the chart shows
DEMO_LIMIT = 150.0   # seconds the demo gets
LOOKS = [100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10_000]   # pulls the rewind stops at
WINDOWS = list(range(1000, PULLS + 1, 1000))   # the last pull of each window of the project demo


# --- the course's code: the testbed and the solutions ------------------------------------------------------------------

_TESTBED = (Path(__file__).parent / "project" / "agent" / "testbed.py").read_text(encoding="utf-8")


def _names(code: str) -> dict:
    out = {"__name__": "__course__"}
    exec(compile(code, "<course>", "exec"), out)
    return out


DriftingBandit = _names(_TESTBED)["DriftingBandit"]

# exercise 1: BanditAgent
_A_HEAD = ("import numpy as np\n\n\n"
           "class BanditAgent:\n"
           "    def __init__(self, k, epsilon, eta, start, rng):\n"
           "        # what the agent knows: an estimate and a count for every machine\n"
           "        self.epsilon = epsilon\n"
           "        self.eta = eta   # None: the learning rate 1/n, the sample average\n"
           "        self.rng = rng\n"
           "        self.q = [start] * k\n"
           "        self.counts = [0] * k\n\n"
           "    def act(self):\n"
           "        # epsilon-greedy: with probability epsilon a random machine,\n"
           "        if self.rng.random() < self.epsilon:\n"
           "            return int(self.rng.integers(0, len(self.q)))\n"
           "        # else the machine with the largest estimate, the first one when several are equal\n")
_A_ACT = "        return int(np.argmax(self.q))\n"
_A_LEARN_HEAD = ("\n    def learn(self, action, reward):\n"
                 "        # one more reward of this machine: count it, take the learning rate\n"
                 "        # (1/n when eta is None, else eta), and move the estimate toward the reward\n")
_A_COUNT = "        self.counts[action] = self.counts[action] + 1\n"
_A_RATE = ("        if self.eta is None:\n"
           "            eta = 1 / self.counts[action]\n"
           "        else:\n"
           "            eta = self.eta\n")
_A_UPDATE = "        self.q[action] = self.q[action] + eta * (reward - self.q[action])\n"
_A_START = _A_HEAD + "        ...\n" + _A_LEARN_HEAD + "        ...\n"


def _a(act: str = _A_ACT, count: str = _A_COUNT, rate: str = _A_RATE, update: str = _A_UPDATE) -> str:
    return _A_HEAD + act + _A_LEARN_HEAD + count + rate + update


_A = _a()

# exercise 2: run
_R_HEAD = ("def run(agent, bandit, pulls):\n"
           "    # the agent and the bandit take turns: the agent chooses a machine, the bandit pays, the agent learns;\n"
           "    # best gets 1 when the agent chose the best machine of that moment, else 0\n"
           "    rewards = []\n"
           "    best = []\n"
           "    for t in range(pulls):\n")
_R_ACT = "        action = agent.act()\n"
_R_BEST = ("        if action == bandit.best():\n"
           "            best.append(1)\n"
           "        else:\n"
           "            best.append(0)\n")
_R_PULL = "        reward = bandit.pull(action)\n"
_R_LEARN = "        agent.learn(action, reward)\n"
_R_KEEP = "        rewards.append(reward)\n"
_R_TAIL = "    return rewards, best\n"
_R_START = _R_HEAD + "        ...\n" + _R_TAIL


def _r(body: str | None = None) -> str:
    return _R_HEAD + (body if body is not None else _R_ACT + _R_BEST + _R_PULL + _R_LEARN + _R_KEEP) + _R_TAIL


_R = _r()

# exercise 3: average
_V_HEAD = ("def average(runs):\n"
           "    # runs: one list for every run, with a number for every pull;\n"
           "    # gives a list with the mean over the runs at every pull\n"
           "    means = []\n"
           "    for t in range(len(runs[0])):\n")
_V_BODY = ("        total = 0\n"
           "        for one in runs:\n"
           "            total = total + one[t]\n"
           "        means.append(total / len(runs))\n")
_V_TAIL = "    return means\n"
_V_START = _V_HEAD + "        ...\n" + _V_TAIL
_V = _V_HEAD + _V_BODY + _V_TAIL

_COURSE = _names(_A + "\n\n" + _R + "\n\n" + _V)
BanditAgent, run, average = _COURSE["BanditAgent"], _COURSE["run"], _COURSE["average"]


def _number(x) -> bool:
    if isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)


# --- the demo: three versions of the learner's agent on the same 200 drifting bandits -----------------------------------

class _Broken(Exception):
    def __init__(self, what: str):
        super().__init__(what)
        self.what = what


def _numbers(v, n: int) -> list | None:
    """v as a list of n finite floats, or None."""
    try:
        out = [float(x.item() if isinstance(x, np.ndarray) else x) for x in v]
    except Exception:   # noqa: BLE001
        return None
    return out if len(out) == n and all(math.isfinite(x) for x in out) else None


def _knows(agent) -> dict | None:
    """What an agent knows: its estimates and counts as lists of K numbers, or None when they cannot be read."""
    q, n = _numbers(getattr(agent, "q", []), K), _numbers(getattr(agent, "counts", []), K)
    return {"q": q, "counts": [int(c) for c in n]} if q is not None and n is not None else None


def train(agent_class=None, run_fn=None, bandits: int = DEMO_BANDITS, pulls: int = PULLS, show=None) -> dict:
    """The demo: for each of the three agents, `bandits` agents made with `agent_class` play the same `bandits` drifting
    bandits, `BLOCK` pulls at a time with `run_fn`. After every block, `show(pull, means, seen)` gets the mean reward
    over the bandits at each pull of the block for each agent, and what the agents know of the watched bandit. Gives the
    rewards and the best choices of every run, as arrays of shape (bandits, pulls). Raises _Broken("run") when
    run_fn gives something other than two lists of BLOCK numbers."""
    agent_class, run_fn = agent_class or BanditAgent, run_fn or run
    agents = {name: [agent_class(K, *SETTINGS[name], np.random.default_rng(AGENT_SEED + b)) for b in range(bandits)]
              for name in AGENTS}
    envs = {name: [DriftingBandit(K, DRIFT, np.random.default_rng(BANDIT_SEED + b)) for b in range(bandits)]
            for name in AGENTS}
    rewards = {name: np.zeros((bandits, pulls), np.float32) for name in AGENTS}
    best = {name: np.zeros((bandits, pulls), np.int8) for name in AGENTS}
    w = min(WATCHED, bandits - 1)   # a short run of a test may have fewer bandits
    looks = {}
    for lo in range(0, pulls, BLOCK):
        hi = min(lo + BLOCK, pulls)
        for name in AGENTS:
            for b in range(bandits):
                try:
                    got = run_fn(agents[name][b], envs[name][b], hi - lo)
                except Timeout:
                    raise
                except Exception as e:   # noqa: BLE001   the demo says which part stopped
                    raise _Broken("stopped") from e
                if not (isinstance(got, (tuple, list)) and len(got) == 2):
                    raise _Broken("run")
                r, h = _numbers(got[0], hi - lo), _numbers(got[1], hi - lo)
                if r is None or h is None:
                    raise _Broken("run")
                rewards[name][b, lo:hi] = r
                best[name][b, lo:hi] = [1 if x else 0 for x in h]
        seen = {"means": [float(m) for m in envs["epsilon"][w].means],
                **{name: _knows(agents[name][w]) for name in AGENTS}}
        if hi in LOOKS:
            looks[hi] = seen
        if show is not None:
            show(hi, {name: rewards[name][:, lo:hi].mean(axis=0).tolist() for name in AGENTS}, seen)
    return {"rewards": rewards, "best": best, "looks": looks, "bandits": bandits, "pulls": pulls}


def project_parts():
    """BanditAgent of agent/bandit.py and run of agent/run.py when they are there and work, else the course's: a dict
    with the class, the function and which of the two are yours; or the name of a file of yours that cannot be used."""
    out = {"agent": BanditAgent, "run": run, "yours": []}
    if projects.state("agent", "bandit") == "yours":
        got = run_code("from agent.bandit import BanditAgent")
        if got.error is None and callable(got.names.get("BanditAgent")):
            out["agent"] = got.names["BanditAgent"]
            out["yours"].append("bandit")
    if (projects.folder() / "agent" / "run.py").exists():
        got = run_code("from agent.run import run")
        if got.error is not None or not callable(got.names.get("run")):
            return "run"
        out["run"] = got.names["run"]
        out["yours"].append("run")
    return out


def demo(show=None):
    """The demo within its time limit: the run with the parts of your project (project_parts), or a string: the file
    that cannot be used, "stopped" when your code stops with an error during the run, "wrong" when run gives back
    something other than two lists, or "slow"."""
    parts = project_parts()
    if isinstance(parts, str):
        return parts
    try:
        out = with_time_limit(lambda: train(parts["agent"], parts["run"], show=show), DEMO_LIMIT)
    except Timeout:
        return "slow"
    except _Broken as e:
        return "wrong" if e.what == "run" else "stopped"
    out["yours"] = parts["yours"]
    return out


def course_demo() -> dict:
    """The demo with the course's agent and run: the numbers the lesson's texts give."""
    return kept("rl-project-bandit:demo", train)


MANY = 2000        # bandits of the course's second run of the demo's three agents
MANY_SEED = 5


def _many() -> dict:
    """The three agents of the demo on MANY other drifting bandits, all at once with NumPy, each agent with its own
    np.random.default_rng(MANY_SEED): per agent, the mean reward and the share of best choices over all pulls and over
    the last 1,000, and over the last 1,000 what a miss costs on average (the mean of the best machine of that moment
    minus the mean of the chosen one) and the part of all that loss that comes from random pulls."""
    out = {}
    rows = np.arange(MANY)
    for name in AGENTS:
        epsilon, eta, start = SETTINGS[name]
        rng = np.random.default_rng(MANY_SEED)
        means = rng.normal(0.0, 1.0, (MANY, K))
        q = np.full((MANY, K), start)
        reward_all = best_all = reward_last = best_last = 0.0
        miss_cost, misses, lost, lost_random = 0.0, 0, 0.0, 0.0
        for t in range(PULLS):
            explore = rng.random(MANY) < epsilon
            action = np.where(explore, rng.integers(0, K, MANY), np.argmax(q, axis=1))
            best = np.argmax(means, axis=1)
            reward = means[rows, action] + rng.normal(size=MANY)
            q[rows, action] += eta * (reward - q[rows, action])
            hit = action == best
            reward_all += reward.mean()
            best_all += hit.mean()
            if t >= PULLS - 1000:
                cost = means[rows, best] - means[rows, action]
                reward_last += reward.mean()
                best_last += hit.mean()
                miss_cost += cost[~hit].sum()
                misses += int((~hit).sum())
                lost += cost.sum()
                lost_random += cost[explore].sum()
            means += rng.normal(0.0, DRIFT, (MANY, K))
        out[name] = {"reward": reward_all / PULLS, "best": best_all / PULLS, "reward_last": reward_last / 1000,
                     "best_last": best_last / 1000, "miss_cost": miss_cost / max(misses, 1),
                     "random_part": lost_random / lost if lost else 0.0}
    return out


def many() -> dict:
    """_many, made once: the numbers the lesson's texts give for 2,000 bandits."""
    return kept("rl-project-bandit:many", _many)


def mine(box_text, correct: bool):
    """The learner's average when its check passes, else None: the project demo then uses the course's."""
    if not correct:
        return None
    fn = run_code(box_text, {"np": np}).names.get("average")
    return fn if callable(fn) else None


def smoothed(ys, k: int = 100) -> list:
    """ys as (pull in thousands, mean over the last k pulls) at every 10th pull, for a line of the demo's charts."""
    out, total = [], 0.0
    for i, y in enumerate(ys):
        total += y
        if i >= k:
            total -= ys[i - k]
        if (i + 1) % 10 == 0:
            out.append(((i + 1) / 1000, total / min(i + 1, k)))
    return out


def mean_of(values, first: int, last: int) -> float:
    """The mean of values[first - 1 : last]: from pull `first` up to and including pull `last`."""
    return float(np.mean(np.asarray(values, dtype=float)[first - 1: last]))


# --- the words and the parts of the pictures ---------------------------------------------------------------------------

LANGS = ("en", "nl")
_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _dec(x: float, lang: str, digits: int = 2) -> str:
    out = f"{x:,.{digits}f}"
    if out in ("-0." + "0" * digits, "-0"):
        out = out[1:]
    out = out.replace("-", "−")
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def _pct(x: float, lang: str, digits: int = 1) -> str:
    out = f"{100 * x:.{digits}f}%"
    return out.replace(".", ",") if lang == "nl" else out


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma in Dutch, and a real minus sign."""
    return _dec(x, lang if lang in LANGS else "en", digits)


def percent(x: float, lang: str) -> str:
    return _pct(x, lang if lang in LANGS else "en")


def _svg(panels: list, alt: str, lang: str, width: int = 330, height: int = 250) -> str:
    out = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(out) if lang == "nl" else out


_WORDS = {
    "en": {"machine": "machine", "reward": "reward", "pull": "pull", "pulls": "pull, in thousands",
           "mean": "mean reward", "share": "share of bandits",
           "greedy": "greedy", "epsilon": "ε-greedy", "optimistic": "optimistic"},
    "nl": {"machine": "automaat", "reward": "reward", "pull": "trek", "pulls": "trek, in duizendtallen",
           "mean": "gemiddelde reward", "share": "deel van de bandits",
           "greedy": "greedy", "epsilon": "ε-greedy", "optimistic": "optimistisch"},
}
# how each agent is drawn in a chart with all three: role, width, dashed; the key under the chart says which is which
_STYLE = {"epsilon": ("ink", 2.4, False), "optimistic": ("ink", 1.4, True), "greedy": ("ink-3", 1.4, True)}


def _key(lang: str) -> str:
    w = _WORDS[lang]
    return live.key_html([(_STYLE[name][0], _STYLE[name][2], w[name]) for name in ("epsilon", "optimistic", "greedy")])


def _bandit_panel(means, knows, title: str, lang: str, low: float = -3.5, high: float = 4.5) -> chart.Panel:
    """One bandit: the mean of every machine as a short line, the agent's estimate as an orange dot, open for a machine
    it never pulled, and above each machine its number of pulls, the odd machines a row lower, so that numbers of four
    digits stay apart. Without what the agent knows (None), the means alone."""
    w = _WORDS[lang]
    p = chart.Panel((-0.6, K - 0.4), (low, high), title=title, xlabel=w["machine"], ylabel=w["reward"],
                    xticks=list(range(K)))
    for a in range(K):
        p.line([a - 0.34, a + 0.34], [means[a], means[a]], "ink", 2.2)
    if knows is None:
        return p
    q, counts = knows["q"], knows["counts"]
    tried = [a for a in range(K) if counts[a] > 0]
    never = [a for a in range(K) if counts[a] == 0]
    if tried:
        p.points(tried, [max(low, min(high, q[a])) for a in tried], "param", r=4.2)
    if never:
        p.points(never, [max(low, min(high, q[a])) for a in never], "param", r=3.6, hollow=True)
    for a in range(K):
        p.text(a, high - (0.35 if a % 2 == 0 else 1.0), f"{counts[a]}", "ink-2")
    return p


# --- the picture of the idea: one turn of the loop ---------------------------------------------------------------------

IDEA_SEED = 10    # the bandit and the agent of the picture
IDEA_PULL = 54   # the turn the picture shows


def idea_turn() -> dict:
    """An epsilon-greedy agent of the course on one drifting bandit: what it knows before pull IDEA_PULL, the machine it
    chooses, the reward, and what it knows after."""
    agent = BanditAgent(K, EPSILON, ETA, 0.0, np.random.default_rng(AGENT_SEED + IDEA_SEED))
    bandit = DriftingBandit(K, DRIFT, np.random.default_rng(BANDIT_SEED + IDEA_SEED))
    run(agent, bandit, IDEA_PULL - 1)
    before = {"q": list(agent.q), "counts": list(agent.counts)}
    means = [float(m) for m in bandit.means]
    action = agent.act()
    best = bandit.best()
    reward = bandit.pull(action)
    agent.learn(action, reward)
    return {"before": before, "after": {"q": list(agent.q), "counts": list(agent.counts)}, "means": means,
            "action": action, "best": best, "reward": reward}


_IDEA_TEXT = {
    "en": {"title": "One turn of the loop: pull {t} of an ε-greedy agent",
           "alt": "The ten machines of one bandit, each with its mean as a short line and the agent's estimate as an orange dot, with its number of pulls above it. At the chosen machine, an open orange dot is the estimate before the turn, a gray dot the reward, and a filled orange dot the estimate after it. An open orange dot elsewhere is a machine the agent never pulled.",
           "note": "Gray: the reward of this pull. Short lines: the means, which only `best()` sees. Orange: the agent's estimates, open for a machine it never pulled; at the chosen machine open before `learn`, filled after it."},
    "nl": {"title": "Eén beurt van de loop: trek {t} van een ε-greedy agent",
           "alt": "De tien automaten van één bandit, elk met het gemiddelde als kort streepje en de schatting van de agent als oranje stip, met erboven hoe vaak eraan getrokken is. Bij de gekozen automaat is een open oranje stip de oude schatting, een grijze stip de reward en een gevulde oranje stip de nieuwe schatting. Een open oranje stip elders hoort bij een automaat waaraan de agent nooit trok.",
           "note": "Grijs: de reward van deze trek. Korte streepjes: de gemiddelden, die alleen `best()` ziet. Oranje: de schattingen van de agent, open bij een automaat waaraan de agent nooit trok; bij de gekozen automaat open de oude schatting, gevuld de nieuwe schatting na `learn`."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    turn = idea_turn()
    p = _bandit_panel(turn["means"], turn["after"], "", lang)
    a = turn["action"]
    p.line([a, a], [turn["before"]["q"][a], turn["after"]["q"][a]], "param", 1.4)
    p.points([a], [turn["before"]["q"][a]], "param", r=4.2, hollow=True)
    p.points([a], [turn["reward"]], "ink-3", r=4.2)
    title = w["title"].format(t=IDEA_PULL)
    return yours.figure(title, _svg([p], w["alt"], lang, width=560, height=280) + _note(w["note"]), alt=title)


# --- the demo's pictures -----------------------------------------------------------------------------------------------

_LIVE_TEXT = {
    "en": {"title": "Your agents: mean reward", "x": "pull, in thousands", "y": "mean reward",
           "alt": "The mean reward over 200 drifting bandits of the ε-greedy agent, over every 10 pulls so far, thin, and its mean over the last 100 pulls, thick; dashed, the mean over the last 100 pulls of the optimistic agent and, lighter, of the greedy agent.",
           "key": "ε-greedy: thin over every 10 pulls, thick the mean over the last 100 pulls",
           "key_other": ": the mean over the last 100 pulls",
           "progress": "**Pull {t} of 10,000** · three agents on 200 drifting bandits",
           "now": "One of the 200 bandits after pull {t}",
           "now_alt": "Three panels, greedy, ε-greedy and optimistic: the ten machines of one bandit, each with its mean at this pull as a short line, the agent's estimate as an orange dot and its number of pulls above it.",
           "rewind_alt": "The mean reward over 200 drifting bandits of each agent, as its mean over the last 100 pulls over the whole run: ε-greedy solid, optimistic dashed, greedy dashed and lighter, with a vertical line at the chosen pull."},
    "nl": {"title": "Je agents: gemiddelde reward", "x": "trek, in duizendtallen", "y": "gemiddelde reward",
           "alt": "De gemiddelde reward over 200 verschuivende bandits van de ε-greedy agent, over elke 10 keer trekken tot nu toe, dun, en het gemiddelde over de laatste 100 keer trekken, dik; gestreept het gemiddelde over de laatste 100 keer trekken van de optimistische agent en, lichter, van de greedy agent.",
           "key": "ε-greedy: dun over elke 10 keer trekken, dik het gemiddelde over de laatste 100 keer trekken",
           "key_other": ": het gemiddelde over de laatste 100 keer trekken",
           "progress": "**Trek {t} van 10.000** · drie agents op 200 verschuivende bandits",
           "now": "Eén van de 200 bandits na trek {t}",
           "now_alt": "Drie panelen, greedy, ε-greedy en optimistisch: de tien automaten van één bandit, elk met het gemiddelde bij deze trek als kort streepje, de schatting van de agent als oranje stip en erboven hoe vaak eraan getrokken is.",
           "rewind_alt": "De gemiddelde reward over 200 verschuivende bandits van elke agent, als het gemiddelde over de laatste 100 keer trekken over de hele run: ε-greedy doorgetrokken, optimistisch gestreept, greedy gestreept en lichter, met een verticale lijn bij de gekozen trek."},
}


def live_words(lang: str) -> dict:
    return _LIVE_TEXT[lang if lang in LANGS else "en"]


def live_key(lang: str) -> list:
    """Under the live chart, what each line is: the epsilon-greedy agent solid, the two others dashed."""
    lang = lang if lang in LANGS else "en"
    w, words = _LIVE_TEXT[lang], _WORDS[lang]
    return [("ink", False, w["key"]), ("ink", True, words["optimistic"] + w["key_other"]),
            ("ink-3", True, words["greedy"] + w["key_other"])]


def progress(t: int, lang: str) -> str:
    return live_words(lang)["progress"].format(t=_dec(t, lang if lang in LANGS else "en", 0))


def now_picture(seen: dict, pull: int, lang: str) -> str:
    """What the three agents know of the watched bandit after `pull` pulls, beside its means at that pull."""
    lang = lang if lang in LANGS else "en"
    w = _LIVE_TEXT[lang] | _WORDS[lang]
    panels = [_bandit_panel(seen["means"], seen[name], w[name], lang) for name in AGENTS]
    title = w["now"].format(t=_dec(pull, lang, 0))
    return yours.figure(title, _svg(panels, w["now_alt"], lang, width=240, height=230), alt=title)


def per_pull(run_: dict) -> dict:
    """The mean over the bandits at every pull of each agent: (rewards, best), kept in the run."""
    if "per_pull" not in run_:
        run_["per_pull"] = {name: (run_["rewards"][name].mean(axis=0).tolist(), run_["best"][name].mean(axis=0).tolist())
                            for name in AGENTS}
    return run_["per_pull"]


def _curves_panel(curves: dict, which: int, lang: str, pulls: int, low: float, high: float, title: str, ylabel: str,
                  at: int | None = None, window: int | None = None) -> chart.Panel:
    """The three agents, each as its mean over the last 100 pulls of curves[name][which], with a line at pull `at` or
    a light band over the 1,000 pulls up to `window`."""
    p = chart.Panel((0, pulls / 1000), (low, high), title=title, xlabel=_WORDS[lang]["pulls"], ylabel=ylabel)
    if window is not None:
        p.rect((window - 1000) / 1000, low, window / 1000, high, "ink-3")
    for name in ("greedy", "optimistic", "epsilon"):   # the solid line last, on top
        role, width, dashed = _STYLE[name]
        ys = smoothed(curves[name][which])
        p.line([t for t, _ in ys], [y for _, y in ys], role, width, dashed=dashed)
    if at is not None:
        p.line([at / 1000, at / 1000], [low, high], "ink-2", 1.2)
    return p


def curve_picture(run_: dict, lang: str, at: int | None = None) -> str:
    """The mean reward of the three agents over the whole run, with a line at pull `at`, and the key under it."""
    lang = lang if lang in LANGS else "en"
    w = _LIVE_TEXT[lang]
    p = _curves_panel(per_pull(run_), 0, lang, run_["pulls"], -0.5, 2.5, w["title"], w["y"], at=at)
    return _svg([p], w["rewind_alt"], lang, width=330, height=240) + _key(lang)


def rewind_frame(run_: dict, pull: int, lang: str) -> str:
    return live.rewound(curve_picture(run_, lang, at=pull), [now_picture(run_["looks"][pull], pull, lang)], width=330)


_REWIND_TEXT = {
    "en": {"said": "After pull {t}: the best machine of this bandit is machine {b}, with a mean of {m}. Greedy estimates it at {g}, ε-greedy at {e}, the optimistic agent at {o}.",
           "unread": "After pull {t}: the best machine of this bandit is machine {b}, with a mean of {m}. The picture cannot read the estimates of your agents: does `BanditAgent` keep them in `self.q` and its counts in `self.counts`, as lists of 10 numbers?"},
    "nl": {"said": "Na trek {t}: de beste automaat van deze bandit is automaat {b}, met een gemiddelde van {m}. Greedy schat de automaat op {g}, ε-greedy op {e}, de optimistische agent op {o}.",
           "unread": "Na trek {t}: de beste automaat van deze bandit is automaat {b}, met een gemiddelde van {m}. De afbeelding kan de schattingen van je agents niet lezen: houdt `BanditAgent` de schattingen bij in `self.q` en de tellingen in `self.counts`, als lijsten van 10 getallen?"},
}


def rewind_text(run_: dict, pull: int, lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    seen = run_["looks"][pull]
    b = max(range(K), key=lambda a: seen["means"][a])
    values = dict(t=_dec(pull, lang, 0), b=b, m=_dec(seen["means"][b], lang))
    if any(seen[name] is None for name in AGENTS):
        return _REWIND_TEXT[lang]["unread"].format(**values)
    return _REWIND_TEXT[lang]["said"].format(**values, **{name[0]: _dec(seen[name]["q"][b], lang) for name in AGENTS})


# --- what the project can do now: both measures, with your average --------------------------------------------------

AVERAGE_SECONDS = 30.0   # what the learner's average gets for the six curves of the project demo together


def curves(run_: dict, average_fn) -> dict | str:
    """With average_fn, the mean reward and the share of best machines at every pull of each agent, over the bandits of
    the run: {name: (rewards, best)}; "stopped", "wrong" or "slow" when average_fn stops with an error, gives something
    other than a number for every pull, or takes too long."""
    def work():
        out = {}
        for name in AGENTS:
            pair = []
            for data in (run_["rewards"][name], run_["best"][name]):
                try:
                    got = average_fn(data.tolist())
                except Timeout:
                    raise
                except Exception as e:   # noqa: BLE001
                    raise _Broken("stopped") from e
                values = _numbers(got, run_["pulls"])
                if values is None:
                    raise _Broken("wrong")
                pair.append(values)
            out[name] = tuple(pair)
        return out
    import contextlib
    import io
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(work, AVERAGE_SECONDS)
    except Timeout:
        return "slow"
    except _Broken as e:
        return e.what


_WINDOW_TEXT = {
    "en": {"reward": "Mean reward", "best": "How often the best machine",
           "alt": "Two charts over the 10,000 pulls, each with the three agents as their mean over the last 100 pulls: the mean reward over the 200 bandits, and the share of bandits in which the agent chose the best machine. A light band marks the chosen 1,000 pulls.",
           "said": "Pulls {a} to {b}. Mean reward: greedy {gr}, ε-greedy {er}, optimistic {orr}. Best machine: greedy {gb}, ε-greedy {eb}, optimistic {ob}."},
    "nl": {"reward": "Gemiddelde reward", "best": "Hoe vaak de beste automaat",
           "alt": "Twee grafieken over de 10.000 keer trekken, elk met de drie agents, als het gemiddelde over de laatste 100 keer trekken: de gemiddelde reward over de 200 bandits, en het deel van de bandits waarin de agent de beste automaat koos. Een lichte band markeert de gekozen 1.000 keer trekken.",
           "said": "Trek {a} tot en met {b}. Gemiddelde reward: greedy {gr}, ε-greedy {er}, optimistisch {orr}. Beste automaat: greedy {gb}, ε-greedy {eb}, optimistisch {ob}."},
}


def window_frame(run_: dict, curves_: dict, end: int, lang: str) -> tuple:
    """The project demo at the window of 1,000 pulls that ends at `end`: the two charts with the band, and the numbers."""
    lang = lang if lang in LANGS else "en"
    w = _WINDOW_TEXT[lang]
    end = min(WINDOWS, key=lambda v: abs(v - end))
    pulls = run_["pulls"]
    panels = [_curves_panel(curves_, 0, lang, pulls, -0.5, 2.5, w["reward"], _WORDS[lang]["mean"], window=end),
              _curves_panel(curves_, 1, lang, pulls, 0.0, 1.0, w["best"], _WORDS[lang]["share"], window=end)]
    first = end - 999
    vals = {}
    for name in AGENTS:
        vals[name[0] + "r"] = _dec(mean_of(curves_[name][0], first, end), lang)
        vals[name[0] + "b"] = _pct(mean_of(curves_[name][1], first, end), lang)
    said = w["said"].format(a=_dec(first, lang, 0), b=_dec(end, lang, 0), gr=vals["gr"], er=vals["er"], orr=vals["or"],
                            gb=vals["gb"], eb=vals["eb"], ob=vals["ob"])
    return _svg(panels, w["alt"], lang, width=330, height=240) + _key(lang), said


# --- the learner's code in the pictures --------------------------------------------------------------------------------

_STOPPED = object()   # what _call gives for code that stopped with an error
_SLOW = object()      # what _call gives for code that ran out of time
PICTURE_SECONDS = 3   # what the learner's code gets for all the calls of one picture together
PIC_SEED = 7          # the bandit of the pictures of exercises 1 and 2
PIC_PULLS = 1000      # pulls of the picture of exercise 1
RUN_PULLS = 100       # pulls of the picture of exercise 2
PIC_RUNS = 50         # runs of the picture of exercise 3, each of PIC_PULLS pulls
FEW = 5               # the first runs of those, for the jumpy line


def _mine(seen, name: str):
    run_ = seen.mine
    if run_ is None:
        return None
    v = run_.names.get(name)
    return v if callable(v) else None


def _call(make, seconds: float = PICTURE_SECONDS):
    """make() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    import contextlib
    import io
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _pic_bandit():
    return DriftingBandit(K, DRIFT, np.random.default_rng(BANDIT_SEED + PIC_SEED))


def _play(agent_class, name: str) -> dict:
    """An agent of `agent_class` with the settings of `name` on the bandit of the pictures, PIC_PULLS pulls with the
    course's run: the means at the end and what the agent knows (or None)."""
    agent = agent_class(K, *SETTINGS[name], np.random.default_rng(AGENT_SEED + PIC_SEED))
    bandit = _pic_bandit()
    run(agent, bandit, PIC_PULLS)
    return {"means": [float(m) for m in bandit.means], "knows": _knows(agent)}


_AGENT_TEXT = {
    "en": {"title": "Your BanditAgent on one drifting bandit",
           "head": "Two of your agents, ε-greedy and optimistic, each with η = 0.1, play the same drifting bandit for 1,000 pulls with the course's `run`.",
           "alt": "Two panels, ε-greedy and optimistic: the ten machines of one bandit after 1,000 pulls, each with its mean as a short line, the estimate of your agent as an orange dot and its number of pulls above it.",
           "end": "The best machine is now machine {b}, with a mean of {m}. Your ε-greedy agent pulled it {ne} times, your optimistic agent {no} times.",
           "same": "Your agents do exactly what the course's agents do.",
           "differ": "Your agents do not do what the course's agents do: they end with other estimates.",
           "unread": "The picture cannot read the estimates of your agents: does `BanditAgent` keep them in `self.q` and its counts in `self.counts`?",
           "none": "Your class stops with an error, or takes too long, when it plays the bandit, so the picture shows the machines alone.",
           "note": "Short lines: the means of the machines after 1,000 pulls. Orange: the estimates of your agent, open for a machine it never pulled."},
    "nl": {"title": "Je BanditAgent op één verschuivende bandit",
           "head": "Twee van je agents, ε-greedy en optimistisch, elk met η = 0,1, spelen 1.000 keer op dezelfde verschuivende bandit, met de `run` van de course.",
           "alt": "Twee panelen, ε-greedy en optimistisch: de tien automaten van één bandit na 1.000 keer trekken, elk met het gemiddelde als kort streepje, de schatting van je agent als oranje stip en erboven hoe vaak eraan getrokken is.",
           "end": "De beste automaat is nu automaat {b}, met een gemiddelde van {m}. Je ε-greedy agent trok er {ne} keer aan, je optimistische agent {no} keer.",
           "same": "Je agents doen precies wat de agents van de course doen.",
           "differ": "Je agents doen niet wat de agents van de course doen: je agents eindigen met andere schattingen.",
           "unread": "De afbeelding kan de schattingen van je agents niet lezen: houdt `BanditAgent` de schattingen bij in `self.q` en de tellingen in `self.counts`?",
           "none": "Je class stopt met een fout, of duurt te lang, tijdens het spel op de bandit, dus de afbeelding toont alleen de automaten.",
           "note": "Korte streepjes: de gemiddelden van de automaten na 1.000 keer trekken. Oranje: de schattingen van je agent, open bij een automaat waaraan nooit getrokken is."},
}


def agent_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _AGENT_TEXT[lang] | _WORDS[lang]
    cls = _mine(seen, "BanditAgent")
    names = ("epsilon", "optimistic")
    got = _call(lambda: {name: _play(cls, name) for name in names}) if cls is not None else _STOPPED
    if not isinstance(got, dict):
        bandit = _pic_bandit()
        for _ in range(PIC_PULLS):
            bandit.pull(0)   # the means drift the same, whatever the agent pulls
        means = [float(m) for m in bandit.means]
        panels = [_bandit_panel(means, None, w[name], lang) for name in names]
        return yours.figure(w["title"], _note(w["head"]) + _svg(panels, w["alt"], lang, width=300, height=240)
                            + _note(w["none"], _LOSS_UI), alt=w["title"])
    panels = [_bandit_panel(got[name]["means"], got[name]["knows"], w[name], lang) for name in names]
    body = _note(w["head"]) + _svg(panels, w["alt"], lang, width=300, height=240)
    means = got["epsilon"]["means"]
    b = max(range(K), key=lambda a: means[a])
    if any(got[name]["knows"] is None for name in names):
        return yours.figure(w["title"], body + _note(w["unread"], _LOSS_UI), alt=w["title"])
    body += _note(w["end"].format(b=b, m=_dec(means[b], lang), ne=_dec(got["epsilon"]["knows"]["counts"][b], lang, 0),
                                  no=_dec(got["optimistic"]["knows"]["counts"][b], lang, 0)))
    want = {name: _play(BanditAgent, name) for name in names}
    same = all(np.allclose(got[name]["knows"]["q"], want[name]["knows"]["q"]) for name in names)
    body += _note(w["same"]) if same else _note(w["differ"], _LOSS_UI)
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


class _Recorder:
    """An agent that notes every machine the agent inside it chooses."""

    def __init__(self, agent):
        self.agent, self.actions = agent, []

    def act(self):
        action = self.agent.act()
        self.actions.append(action)
        return action

    def learn(self, action, reward):
        self.agent.learn(action, reward)


def best_track(pulls: int = RUN_PULLS) -> list:
    """The best machine of the bandit of the pictures before each pull: its means drift the same, whatever is pulled."""
    bandit, out = _pic_bandit(), []
    for _ in range(pulls):
        out.append(bandit.best())
        bandit.pull(0)
    return out


_RUN_TEXT = {
    "en": {"title": "Your run: the first 100 pulls",
           "head": "The course's ε-greedy agent and the bandit of the picture above take turns in your `run`, 100 pulls.",
           "alt": "100 pulls of one bandit: a gray line steps along the best machine at every pull, and a dot at each pull marks the machine the agent chose, dark when your run marked it as the best machine, purple when not.",
           "end": "Your `run` gave back 100 rewards, with a mean of {r}, and 100 marks: the agent chose the best machine in {k} of the 100 pulls.",
           "off": "Your marks do not match the best machine: at pull {t} the agent chose machine {a}, the best machine was {b}, and your `best` holds {got} there. Do you ask for `bandit.best()` before the pull?",
           "none": "Your `run` stops with an error, takes too long or gives back something other than two lists of 100 numbers here, so the picture shows the best machine alone.",
           "note": "Gray: the best machine at every pull. Dots: the machine the agent chose, dark when your `best` says it was the best machine, purple when not."},
    "nl": {"title": "Je run: de eerste 100 keer trekken",
           "head": "De ε-greedy agent van de course en de bandit uit de afbeelding hierboven zijn om de beurt aan zet in je `run`, 100 keer trekken.",
           "alt": "100 keer trekken op één bandit: een grijze lijn volgt de beste automaat bij elke trek, en een stip bij elke trek markeert de automaat die de agent koos, donker als je run de keuze als beste automaat markeerde, paars als dat niet zo was.",
           "end": "Je `run` gaf 100 rewards terug, met een gemiddelde van {r}, en 100 markeringen: de agent koos bij {k} van de 100 keer trekken de beste automaat.",
           "off": "Je markeringen kloppen niet met de beste automaat: bij trek {t} koos de agent automaat {a}, de beste automaat was {b}, en je `best` heeft daar {got}. Vraag je `bandit.best()` op voordat er getrokken wordt?",
           "none": "Je `run` stopt hier met een fout, duurt te lang of geeft iets anders terug dan twee lijsten van 100 getallen, dus de afbeelding toont alleen de beste automaat.",
           "note": "Grijs: de beste automaat bij elke trek. Stippen: de automaat die de agent koos, donker als je `best` zegt dat het de beste automaat was, paars als dat niet zo was."},
}


def run_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _RUN_TEXT[lang] | _WORDS[lang]
    fn = _mine(seen, "run")
    track = best_track()
    xs = list(range(1, RUN_PULLS + 1))
    p = chart.Panel((0, RUN_PULLS + 1), (-0.6, K - 0.4), title="", xlabel=w["pull"], ylabel=w["machine"],
                    yticks=list(range(K)))
    steps_x, steps_y = [], []
    for t, b in zip(xs, track):
        steps_x += [t - 0.5, t + 0.5]
        steps_y += [b, b]
    p.line(steps_x, steps_y, "ink-3", 2.2)
    head = _note(w["head"])
    got = None
    if fn is not None:
        agent = _Recorder(BanditAgent(K, *SETTINGS["epsilon"], np.random.default_rng(AGENT_SEED + PIC_SEED)))
        out = _call(lambda: fn(agent, _pic_bandit(), RUN_PULLS))
        if isinstance(out, (tuple, list)) and len(out) == 2 and len(agent.actions) == RUN_PULLS:
            r, h = _numbers(out[0], RUN_PULLS), _numbers(out[1], RUN_PULLS)
            if r is not None and h is not None:
                got = (r, h, [int(a) for a in agent.actions])
    if got is None:
        return yours.figure(w["title"], head + _svg([p], w["alt"], lang, width=560, height=260) + _note(w["none"], _LOSS_UI),
                            alt=w["title"])
    r, h, actions = got
    hit = [t for t in range(RUN_PULLS) if h[t] == 1]
    miss = [t for t in range(RUN_PULLS) if h[t] != 1]
    if hit:
        p.points([xs[t] for t in hit], [actions[t] for t in hit], "ink", r=3.2)
    if miss:
        p.points([xs[t] for t in miss], [actions[t] for t in miss], "loss", r=3.2)
    body = head + _svg([p], w["alt"], lang, width=560, height=260)
    body += _note(w["end"].format(r=_dec(float(np.mean(r)), lang), k=len(hit)))
    off = next((t for t in range(RUN_PULLS) if (h[t] == 1) != (actions[t] == track[t])), None)
    if off is not None:
        body += _note(w["off"].format(t=off + 1, a=actions[off], b=track[off], got=_dec(h[off], lang, 0)), _LOSS_UI)
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


def pic_runs() -> list:
    """The best choices of the course's ε-greedy agent on PIC_RUNS drifting bandits of PIC_PULLS pulls: the runs of the
    picture of exercise 3."""
    def make():
        out = []
        for b in range(PIC_RUNS):
            agent = BanditAgent(K, *SETTINGS["epsilon"], np.random.default_rng(AGENT_SEED + 500 + b))
            bandit = DriftingBandit(K, DRIFT, np.random.default_rng(BANDIT_SEED + 500 + b))
            out.append(run(agent, bandit, PIC_PULLS)[1])
        return out
    return kept("rl-project-bandit:pic-runs", make)


_AVERAGE_TEXT = {
    "en": {"title": "Your average over 5 runs and over 50",
           "head": "The course's ε-greedy agent on 50 drifting bandits of 1,000 pulls each. Your `average` of its best choices, over the first 5 runs and over all 50.",
           "alt": "The share of runs in which the agent chose the best machine at every pull of 1,000: a jumpy gray line over 5 runs, and a calmer dark line over 50 runs.",
           "end": "Over the last 500 pulls: {few} with 5 runs, {many} with 50. Over 5 runs, your average can only be 0, 0.2, 0.4, 0.6, 0.8 or 1 at a pull.",
           "none": "Your `average` stops with an error, takes too long or does not give a number for every pull here, so the picture shows nothing yet.",
           "note": "Gray: your average over 5 runs. Dark line: over all 50 runs."},
    "nl": {"title": "Je average over 5 runs en over 50",
           "head": "De ε-greedy agent van de course op 50 verschuivende bandits van elk 1.000 keer trekken. Je `average` van de beste keuzes, over de eerste 5 runs en over alle 50.",
           "alt": "Het deel van de runs waarin de agent de beste automaat koos, bij elk van 1.000 keer trekken: een springerige grijze lijn over 5 runs, en een rustigere donkere lijn over 50 runs.",
           "end": "Over de laatste 500 keer trekken: {few} met 5 runs, {many} met 50. Over 5 runs kan je gemiddelde bij een trek alleen 0, 0,2, 0,4, 0,6, 0,8 of 1 zijn.",
           "none": "Je `average` stopt hier met een fout, duurt te lang of geeft niet bij elke trek een getal, dus de afbeelding toont nog niets.",
           "note": "Grijs: je gemiddelde over 5 runs. Donkere lijn: over alle 50 runs."},
}


def average_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _AVERAGE_TEXT[lang] | _WORDS[lang]
    fn = _mine(seen, "average")
    data = pic_runs()
    got = _call(lambda: (fn([list(r) for r in data[:FEW]]), fn([list(r) for r in data]))) if fn is not None else None
    few = many = None
    if isinstance(got, tuple):
        few, many = _numbers(got[0], PIC_PULLS), _numbers(got[1], PIC_PULLS)
    if few is None or many is None:
        return yours.figure(w["title"], _note(w["head"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    xs = list(range(1, PIC_PULLS + 1))
    p = chart.Panel((0, PIC_PULLS), (0, 1), title="", xlabel=w["pull"], ylabel=w["share"].replace("bandits", "runs"))
    p.line(xs, few, "ink-3", 1.0)
    p.line(xs, many, "ink", 2.2)
    body = _note(w["head"]) + _svg([p], w["alt"], lang, width=560, height=260)
    body += _note(w["end"].format(few=_pct(mean_of(few, 501, PIC_PULLS), lang), many=_pct(mean_of(many, 501, PIC_PULLS), lang)))
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["agent", "run", "average", "measures"]


class _Val:
    """A test value that a check message shows as code: written out, or by a name of the lesson."""

    def __init__(self, shown: str, value):
        self.shown, self.value = shown, value

    def __repr__(self) -> str:
        return self.shown

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


def _plain(v):
    """v with NumPy values and arrays as Python numbers and lists."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return type(v)(_plain(x) for x in v)
    return v


def _close(got, want) -> bool:
    got, want = _plain(got), _plain(want)
    if isinstance(want, (list, tuple)):
        return (isinstance(got, (list, tuple)) and len(got) == len(want)
                and all(_close(g, w) for g, w in zip(got, want)))
    if isinstance(want, bool) or isinstance(got, bool):
        return got == want
    if isinstance(want, numbers.Real):
        return isinstance(got, numbers.Real) and math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-6)
    return got == want


def _same_curve(got, want) -> bool:
    """The means of every pull: a list, tuple or array of numbers, as long as the course's and with the same values."""
    got = _plain(got)
    if not isinstance(got, (list, tuple)) or len(got) != len(want):
        return False
    return all(_number(g) and math.isclose(float(g), float(w), rel_tol=1e-6, abs_tol=1e-6) for g, w in zip(got, want))


def _value(v) -> str:
    """A value the way code writes it, numbers rounded to six decimals."""
    v = _plain(v)
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_value(x) for x in v)
        return f"({inner})" if isinstance(v, tuple) else f"[{inner}]"
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, float):
        return repr(round(v, 6) + 0.0)
    return repr(v)


def _show(v):
    out = _value(v)
    return out if len(out) <= 300 else out[:300] + "…"


def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _kept(name: str) -> dict:
    return {"en": f"Your `{name}` is right. You changed `{{saved}}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
            "nl": f"Je `{name}` klopt. Je hebt `{{saved}}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."}


_SCENARIO = {"en": "With these lines after your {what}:\n\n```python\n{{scenario}}\n```\n\nyou get `{{got}}`, and it should be `{{want}}`",
             "nl": "Met deze regels na je {what}:\n\n```python\n{{scenario}}\n```\n\nkrijg je `{{got}}`, en dat moet `{{want}}` zijn"}


def _scenario(lang: str, what: str) -> str:
    return _SCENARIO[lang].format(what=what)


# exercise 1: BanditAgent
_A_SCENARIOS = [
    # the example: three rewards with learning rate 0.5, then a greedy choice
    "import numpy as np\n\nagent = BanditAgent(3, 0.0, 0.5, 0.0, np.random.default_rng(0))\nagent.learn(1, 2.0)\n"
    "agent.learn(1, 4.0)\nagent.learn(2, 1.0)\nagent.q, agent.counts, agent.act()",
    # eta None: the sample average, 1/n
    "import numpy as np\n\nagent = BanditAgent(2, 0.0, None, 0.0, np.random.default_rng(0))\nfor reward in [3.0, 5.0, 1.0]:\n"
    "    agent.learn(0, reward)\nagent.learn(1, -2.0)\nagent.q, agent.counts, agent.act()",
    # an optimistic start: equal estimates, the first of them, and every reward disappoints
    "import numpy as np\n\nagent = BanditAgent(4, 0.0, 0.1, 5.0, np.random.default_rng(0))\nchosen = []\nfor t in range(6):\n"
    "    action = agent.act()\n    chosen.append(action)\n    agent.learn(action, 1.0)\nchosen, [round(q, 2) for q in agent.q]",
    # exploring half of the time
    "import numpy as np\n\nagent = BanditAgent(3, 0.5, 0.2, 0.0, np.random.default_rng(4))\nchosen = []\nfor t in range(12):\n"
    "    action = agent.act()\n    chosen.append(action)\n    agent.learn(action, [1.0, 2.0, 0.5][action])\n"
    "chosen, [round(q, 3) for q in agent.q], agent.counts",
]
_A_MAX = _a(act="        return max(self.q)\n")
_A_NO_COUNT = _a(count="")
_A_LATE = _a(count="", update=_A_UPDATE + _A_COUNT)
_A_ALWAYS = _a(rate="        eta = 1 / self.counts[action]\n")
_A_NONE = _a(rate="")
_A_NONE = _A_NONE.replace("eta * (reward", "self.eta * (reward")
_A_AWAY = _a(update="        self.q[action] = self.q[action] + eta * (self.q[action] - reward)\n")
_A_JUMP = _a(update="        self.q[action] = reward\n")

# exercise 2: run
_R_IMPORTS = "import numpy as np\nfrom agent.bandit import BanditAgent\nfrom agent.testbed import DriftingBandit\n\n"
_R_SCENARIOS = [
    # the example: a greedy agent with learning rate 0.5 on a bandit that does not drift, 4 pulls
    _R_IMPORTS + "agent = BanditAgent(3, 0.0, 0.5, 0.0, np.random.default_rng(1))\n"
    "bandit = DriftingBandit(3, 0.0, np.random.default_rng(2))\nrewards, best = run(agent, bandit, 4)\n"
    "[round(r, 3) for r in rewards], best, [round(q, 3) for q in agent.q]",
    # a second call goes on where the first stopped: the objects keep what they know
    _R_IMPORTS + "agent = BanditAgent(3, 0.0, 0.5, 0.0, np.random.default_rng(1))\n"
    "bandit = DriftingBandit(3, 0.0, np.random.default_rng(2))\nrun(agent, bandit, 3)\nrewards, best = run(agent, bandit, 3)\n"
    "best, agent.counts",
    # means that drift a lot: the best machine of each pull is the one before the pull
    _R_IMPORTS + "agent = BanditAgent(4, 0.2, 0.1, 0.0, np.random.default_rng(3))\n"
    "bandit = DriftingBandit(4, 1.0, np.random.default_rng(4))\nrewards, best = run(agent, bandit, 30)\n"
    "best, [round(r, 3) for r in rewards[:5]]",
    # no pulls at all
    _R_IMPORTS + "agent = BanditAgent(3, 0.1, None, 0.0, np.random.default_rng(5))\n"
    "bandit = DriftingBandit(3, 0.01, np.random.default_rng(6))\nrun(agent, bandit, 0)",
]
_R_AFTER = _r(_R_ACT + _R_PULL + _R_BEST + _R_LEARN + _R_KEEP)
_R_NO_LEARN = _r(_R_ACT + _R_BEST + _R_PULL + _R_KEEP)
_R_TWICE = _r("        if agent.act() == bandit.best():\n            best.append(1)\n        else:\n            best.append(0)\n"
              "        action = agent.act()\n" + _R_PULL + _R_LEARN + _R_KEEP)
_R_PULL_BEST = _r(_R_ACT + _R_BEST + "        reward = bandit.pull(bandit.best())\n" + _R_LEARN + _R_KEEP)
_R_ORDER = _r(_R_ACT + _R_BEST + _R_PULL + "        agent.learn(reward, action)\n" + _R_KEEP)
_R_NO_RETURN = _R_HEAD + _R_ACT + _R_BEST + _R_PULL + _R_LEARN + _R_KEEP

# exercise 3: average
SMALL_RUNS = [[0, 1, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]]   # three runs of four pulls: was the best machine chosen?
AVERAGE_CASES = [(_Val("small_runs", SMALL_RUNS),), ([[1.5, -0.5], [0.5, 2.5]],), ([[2.0, 4.0, 6.0]],),
                 ([[1, 0], [1, 1], [0, 0], [1, 0]],), ([[0.2, 0.4, 0.6, 0.8, 1.0], [1.0, 0.8, 0.6, 0.4, 0.2], [0.3, 0.3, 0.3, 0.3, 0.3]],)]
_V_PULLS = _V.replace("total / len(runs)", "total / len(runs[0])")
_V_OUTSIDE = _V_HEAD.replace("    means = []\n", "    means = []\n    total = 0\n") + _V_BODY.replace("        total = 0\n", "") + _V_TAIL
_V_NO_DIVIDE = _V.replace("means.append(total / len(runs))", "means.append(total)")
_V_NO_RETURN = _V_HEAD + _V_BODY

_SA, _SA_NL = _scenario("en", "class"), _scenario("nl", "class")
_SR, _SR_NL = _scenario("en", "code"), _scenario("nl", "code")

EXERCISES = {
    "agent": ClassBox(
        name="agent",
        picture=agent_picture,
        start=_A_START,
        solution=_A,
        scenarios=_A_SCENARIOS,
        compare=_close,
        first=[_DOTS],
        save_as=("agent", "bandit"),
        mistakes=[
            Mistake("zero-count", error=lambda e: isinstance(e, ZeroDivisionError),
                    en="Python divides by 0 when `eta` is None: the count of the machine is still 0 at `1 / self.counts[action]`. A reward is the first of its machine when the count goes from 0 to 1; do you add 1 to `self.counts[action]` before you divide?",
                    nl="Python deelt door 0 als `eta` None is: de telling van de automaat is nog 0 bij `1 / self.counts[action]`. Een reward is de eerste van de automaat als de telling van 0 naar 1 gaat; tel je 1 op bij `self.counts[action]` voordat je deelt?"),
            Mistake("none-eta", error=lambda e: isinstance(e, TypeError) and "NoneType" in str(e) and "unsupported operand" in str(e),
                    en="Python cannot multiply by `None`: with `eta` None, your `learn` uses `self.eta` as the learning rate itself. None means 1/n, the sample average; can you give a name such as `eta` the value `1 / self.counts[action]` when `self.eta is None`, and `self.eta` otherwise?",
                    nl="Python kan niet met `None` vermenigvuldigen: met `eta` None gebruikt je `learn` `self.eta` zelf als learning rate. None betekent 1/n, het steekproefgemiddelde; kun je een naam zoals `eta` de waarde `1 / self.counts[action]` geven als `self.eta is None`, en anders `self.eta`?"),
            Mistake("max-value", wrong=_A_MAX,
                    en=_SA + ": `act` gives back the largest estimate itself, 2.5, and not the machine that has it, 1. Which NumPy function gives the place of the largest value?",
                    nl=_SA_NL + ": `act` geeft de grootste schatting zelf terug, 2,5, en niet de automaat die de schatting heeft, 1. Welke NumPy-functie geeft de plaats van de grootste waarde?"),
            Mistake("no-count", wrong=_A_NO_COUNT,
                    en=_SA + ": the counts stay 0. Each reward is one more of its machine; where does `learn` add 1 to `self.counts[action]`?",
                    nl=_SA_NL + ": de tellingen blijven 0. Elke reward is er één meer van de automaat; waar telt `learn` 1 op bij `self.counts[action]`?"),
            Mistake("always-average", wrong=_A_ALWAYS,
                    en=_SA + ": your agent always takes 1/n, even when `eta` is 0.5. 1/n is for `eta` None only; does `learn` use `self.eta` when it is a number?",
                    nl=_SA_NL + ": je agent neemt altijd 1/n, ook als `eta` 0,5 is. 1/n hoort alleen bij `eta` None; gebruikt `learn` `self.eta` als het een getal is?"),
            Mistake("away", wrong=_A_AWAY,
                    en=_SA + ": the estimate moves away from the reward. The reward 2.0 lies above the estimate 0.0; in which order do you subtract, so that the step goes up?",
                    nl=_SA_NL + ": de schatting gaat van de reward af. De reward 2,0 ligt boven de schatting 0,0; in welke volgorde trek je af, zodat de stap omhoog gaat?"),
            Mistake("jump", wrong=_A_JUMP,
                    en=_SA + ": the estimate jumps all the way to the reward, so it forgets every reward before. How big a part of the way does the learning rate say to go?",
                    nl=_SA_NL + ": de schatting springt helemaal naar de reward, dus vergeet de schatting alle eerdere rewards. Welk deel van de weg zegt de learning rate?"),
        ],
        texts={
            **_SAME_TEXTS,
            "empty": {"en": "Replace the two `...`: in `act` the machine with the largest estimate, in `learn` the count, the learning rate and the step of the estimate toward the reward.",
                      "nl": "Vervang de twee `...`: in `act` de automaat met de grootste schatting, in `learn` de telling, de learning rate en de stap van de schatting naar de reward."},
            "wrong": {"en": _SA + ". Does `act` give back the place of the largest estimate, and does `learn` count the reward, take 1/n or `self.eta`, and move `self.q[action]` toward the reward?",
                      "nl": _SA_NL + ". Geeft `act` de plaats van de grootste schatting terug, en telt `learn` de reward, neemt `learn` 1/n of `self.eta`, en schuift `self.q[action]` op naar de reward?"},
            "almost": {"en": "Right for the example. But " + _SA[0].lower() + _SA[1:] + ". Does your class also work with `eta` None, with several equal estimates, and while it explores?",
                       "nl": "Goed voor het voorbeeld. Maar " + _SA_NL[0].lower() + _SA_NL[1:] + ". Werkt je class ook met `eta` None, met meerdere gelijke schattingen, en terwijl de agent verkent?"},
            "correct": {"en": "Right: after the rewards 2.0 and 4.0, machine 1 has the estimate 2.5, and `act` chooses it. Your `BanditAgent` is now in your project, as `{saved}`. In the picture, two of your agents play one drifting bandit; which of them pulled the best machine more often?",
                        "nl": "Goed: na de rewards 2,0 en 4,0 heeft automaat 1 de schatting 2,5, en `act` kiest de automaat. Je `BanditAgent` staat nu in je project, als `{saved}`. In de afbeelding spelen twee van je agents op één verschuivende bandit; welke van de twee trok vaker aan de beste automaat?"},
            "kept": _kept("BanditAgent"),
        },
    ),
    "run": ScenarioBox(
        name="run",
        picture=run_picture,
        start=_R_START,
        solution=_R,
        scenarios=_R_SCENARIOS,
        compare=_close,
        first=[_DOTS],
        save_as=("agent", "run"),
        mistakes=[
            Mistake("no-testbed", error=lambda e: isinstance(e, ImportError) and "testbed" in str(e),
                    en="Python cannot import `DriftingBandit`. The check imports it with `from agent.testbed import DriftingBandit`; does your code import it in a different way? If not, the file `work/projects/agent/testbed.py` that the course put in your project is gone or changed. The course puts the file back when you open the lesson again while it is gone; can you delete a changed one and open the lesson again?",
                    nl="Python kan `DriftingBandit` niet importeren. De check importeert de class met `from agent.testbed import DriftingBandit`; importeert je code de class op een andere manier? Zo niet, dan is het bestand `work/projects/agent/testbed.py` van de course weg of veranderd. De course zet het bestand terug als je de les opnieuw opent terwijl het bestand weg is; kun je een veranderd bestand verwijderen en de les opnieuw openen?"),
            Mistake("learn-order", error=lambda e: isinstance(e, TypeError) and "indices must be integers" in str(e),
                    en="Python stops in `learn`: it uses a reward as the place of a machine in a list. `learn` takes the machine first and then the reward, `agent.learn(action, reward)`; in which order does your loop give them?",
                    nl="Python stopt in `learn`: `learn` gebruikt een reward als de plaats van een automaat in een lijst. `learn` krijgt eerst de automaat en dan de reward, `agent.learn(action, reward)`; in welke volgorde geeft je loop de twee mee?"),
            Mistake("no-return", error=lambda e: isinstance(e, TypeError) and "NoneType" in str(e) and "unpack" in str(e),
                    en="Your `run` gives back nothing, `None`, so the lines after it cannot take `rewards, best` from it. Does it end with `return rewards, best`, after the loop?",
                    nl="Je `run` geeft niets terug, `None`, dus de regels erna kunnen er geen `rewards, best` uit halen. Eindigt je `run` met `return rewards, best`, na de loop?"),
            Mistake("best-after", wrong=_R_AFTER,
                    en=_SR + ": your loop asks for the best machine after the pull, and `pull` lets every mean drift. Which machine was the best when the agent chose? Do you ask `bandit.best()` before `bandit.pull(action)`?",
                    nl=_SR_NL + ": je loop vraagt de beste automaat op na de trek, en `pull` laat elk gemiddelde verschuiven. Welke automaat was de beste toen de agent koos? Vraag je `bandit.best()` op voordat `bandit.pull(action)` komt?"),
            Mistake("no-learn", wrong=_R_NO_LEARN,
                    en=_SR + ": your agent never learns, so it keeps choosing machine 0 and its estimates stay 0. After the pull the agent gets the reward; which method of the agent takes it?",
                    nl=_SR_NL + ": je agent leert nooit, dus blijft de agent automaat 0 kiezen en blijven de schattingen 0. Na de trek krijgt de agent de reward; welke method van de agent neemt de reward aan?"),
            Mistake("two-acts", wrong=_R_TWICE,
                    en=_SR + ": your loop asks the agent to choose twice in one pull, so `best` is about a different choice from the machine that is pulled. Can you keep the one choice in `action` and use it for both?",
                    nl=_SR_NL + ": je loop laat de agent twee keer kiezen bij één trek, dus gaat `best` over een andere keuze dan de automaat waaraan getrokken wordt. Kun je de ene keuze in `action` bewaren en voor allebei gebruiken?"),
            Mistake("pull-best", wrong=_R_PULL_BEST,
                    en=_SR + ": the bandit pays for the best machine, not for the machine the agent chose. `best()` is only for measuring, the agent never sees it; which machine goes into `bandit.pull(...)`?",
                    nl=_SR_NL + ": de bandit betaalt voor de beste automaat, niet voor de automaat die de agent koos. `best()` is alleen om te meten, de agent ziet het nooit; welke automaat gaat er in `bandit.pull(...)`?"),
        ],
        texts={
            **_SAME_TEXTS,
            "empty": {"en": "Replace the `...` with one turn: the agent chooses, `best` gets 1 or 0, the bandit pays, the agent learns, and `rewards` keeps the reward.",
                      "nl": "Vervang de `...` door één beurt: de agent kiest, `best` krijgt 1 of 0, de bandit betaalt, de agent leert, en `rewards` bewaart de reward."},
            "wrong": {"en": _SR + ". Does each turn choose once, ask `bandit.best()` before the pull, pull the chosen machine and give the reward to `learn`?",
                      "nl": _SR_NL + ". Kiest elke beurt één keer, vraag je `bandit.best()` op voordat de trek komt, trek je aan de gekozen automaat en geef je de reward aan `learn`?"},
            "almost": {"en": "Right for the example. But " + _SR[0].lower() + _SR[1:] + ". Does your loop work for any number of pulls, and for means that drift?",
                       "nl": "Goed voor het voorbeeld. Maar " + _SR_NL[0].lower() + _SR_NL[1:] + ". Werkt je loop bij elk aantal keer trekken, en bij gemiddelden die verschuiven?"},
            "correct": {"en": "Right: four turns, and every reward went to the agent. Your `run` is now in your project, as `{saved}`. In the picture it lets an ε-greedy agent play 100 pulls; at which pulls did the agent miss the best machine?",
                        "nl": "Goed: vier beurten, en elke reward ging naar de agent. Je `run` staat nu in je project, als `{saved}`. In de afbeelding laat je `run` een ε-greedy agent 100 keer trekken; bij welke keren miste de agent de beste automaat?"},
            "kept": _kept("run"),
        },
    ),
    "average": FunctionBox(
        name="average",
        picture=average_picture,
        function="average",
        names={"np": np},
        start=_V_START,
        solution=_V,
        show=_show,
        compare=_same_curve,
        visible=AVERAGE_CASES[:1],
        hidden=AVERAGE_CASES[1:],
        first=[_DOTS],
        save_as=("agent", "measure"),
        mistakes=[
            Mistake("index", error=lambda e: isinstance(e, IndexError),
                    en="Python stops: a place in a list goes past its end. `runs` has one list for every run, and `t` counts the pulls: pull `t` of the run `one` is `one[t]`, while `runs[t]` is a whole run. Which of the two do you add?",
                    nl="Python stopt: een plaats in een lijst gaat voorbij het eind. `runs` heeft één lijst voor elke run, en `t` telt de keren trekken: trek `t` van de run `one` is `one[t]`, terwijl `runs[t]` een hele run is. Welke van de twee tel je op?"),
            Mistake("pulls", wrong=_V_PULLS,
                    en="`{call}` gives {got}, and it should give {want}: you divide by the number of pulls of a run, 4. The mean at one pull is over the runs; how many runs are there?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deelt door het aantal keer trekken van een run, 4. Het gemiddelde bij één trek gaat over de runs; hoeveel runs zijn er?"),
            Mistake("outside", wrong=_V_OUTSIDE,
                    en="`{call}` gives {got}, and it should give {want}: `total` starts at 0 once, before all pulls, so every pull adds to the totals of the pulls before. Where should `total = 0` go, so that each pull starts again?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `total` begint één keer bij 0, voor alle keren trekken samen, dus telt elke trek op bij de totalen van de eerdere keren. Waar hoort `total = 0`, zodat elke trek opnieuw begint?"),
            Mistake("no-divide", wrong=_V_NO_DIVIDE,
                    en="`{call}` gives {got}, and it should give {want}: those are the totals over the runs. A mean is the total divided by how many numbers it adds; what do you divide by?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: dat zijn de totalen over de runs. Een gemiddelde is het totaal gedeeld door hoeveel getallen het optelt; waardoor deel je?"),
        ],
        texts={
            **_SAME_TEXTS,
            "empty": {"en": "Replace the `...` with the mean at pull `t`: add `one[t]` of every run `one` in `runs`, divide by the number of runs, and append it to `means`.",
                      "nl": "Vervang de `...` door het gemiddelde bij trek `t`: tel `one[t]` van elke run `one` in `runs` op, deel door het aantal runs, en voeg het toe aan `means`."},
            "arguments": {"en": "The check calls `{call}`, with one value, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(runs)`?",
                          "nl": "De check roept `{call}` aan, met één waarde, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(runs)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is each mean the total of `one[t]` over the runs, divided by the number of runs?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is elk gemiddelde het totaal van `one[t]` over de runs, gedeeld door het aantal runs?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any number of runs and pulls?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie bij elk aantal runs en keren trekken?"},
            "no_return": {"en": "With `{call}`, your function gives back `None`, and it should give a list. Does it end with `return means`?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en het moet een lijst zijn. Eindigt je functie met `return means`?"},
            "correct": {"en": "Right: at pull 3, two of the three runs chose the best machine, so the mean is 2/3. Your `average` is now in your project, as `{saved}`. In the picture it averages 5 runs and 50; which line can you trust more?",
                        "nl": "Goed: bij trek 3 kozen twee van de drie runs de beste automaat, dus is het gemiddelde 2/3. Je `average` staat nu in je project, als `{saved}`. In de afbeelding middelt je `average` over 5 runs en over 50; welke lijn kun je meer vertrouwen?"},
            "kept": _kept("average"),
        },
    ),
}

CASES = [
    ("agent", _A_START, "empty", None),
    ("agent", _A, "correct", None),
    ("agent", _a(act="        ...\n"), "wrong", "dots-left"),
    ("agent", _A_LATE, "error", "zero-count"),
    ("agent", _A_NONE, "error", "none-eta"),
    ("agent", _A_MAX, "wrong", "max-value"),
    ("agent", _A_NO_COUNT, "wrong", "no-count"),
    ("agent", _A_ALWAYS, "wrong", "always-average"),
    ("agent", _A_AWAY, "wrong", "away"),
    ("agent", _A_JUMP, "wrong", "jump"),
    ("agent", _a(act="        return self.q.index(max(self.q))\n"), "correct", None),
    ("agent", _a(rate="        eta = self.eta\n        if eta is None:\n            eta = 1 / self.counts[action]\n",
                 update="        self.q[action] += eta * (reward - self.q[action])\n"), "correct", None),
    ("agent", _a(act="        return int(np.argmin(self.q))\n"), "wrong", None),
    ("agent", _a(rate="        if self.eta is None:\n            eta = 1 / self.counts[action]\n        else:\n            eta = self.eta\n"
                 "        if reward < 0:\n            eta = 0.0\n"), "almost", None),
    ("run", _R_START, "empty", None),
    ("run", _R, "correct", None),
    ("run", _R_HEAD + "        action = agent.act()\n        ...\n" + _R_TAIL, "wrong", "dots-left"),
    ("run", "from testbed import DriftingBandit\n\n" + _R, "error", "no-testbed"),
    ("run", _R_ORDER, "error", "learn-order"),
    ("run", _R_NO_RETURN, "error", "no-return"),
    ("run", _R_AFTER, "wrong", "best-after"),
    ("run", _R_NO_LEARN, "wrong", "no-learn"),
    ("run", _R_TWICE, "wrong", "two-acts"),
    ("run", _R_PULL_BEST, "wrong", "pull-best"),
    ("run", _r(_R_ACT + "        best.append(int(action == bandit.best()))\n" + _R_PULL + _R_LEARN + _R_KEEP), "correct", None),
    ("run", _r(_R_ACT + "        best.append(action == bandit.best())\n        reward = bandit.pull(action)\n"
               "        rewards.append(reward)\n        agent.learn(action, reward)\n"), "correct", None),
    ("run", _r(_R_ACT + _R_BEST + _R_PULL + _R_LEARN + "        rewards.append(round(reward, 1))\n"), "wrong", None),
    ("run", _r(_R_ACT + _R_BEST + _R_PULL + _R_LEARN + _R_KEEP + "        if t == 3:\n            break\n"), "almost", None),
    ("average", _V_START, "empty", None),
    ("average", _V, "correct", None),
    ("average", _V_HEAD + "        total = 0\n        ...\n" + _V_TAIL, "wrong", "dots-left"),
    ("average", _V_HEAD + "        means.append(np.mean(runs[t]))\n" + _V_TAIL, "error", "index"),
    ("average", _V_PULLS, "wrong", "pulls"),
    ("average", _V_OUTSIDE, "wrong", "outside"),
    ("average", _V_NO_DIVIDE, "wrong", "no-divide"),
    ("average", _V_NO_RETURN, "wrong", "no-return"),
    ("average", _V_HEAD + "        means.append(np.mean([one[t] for one in runs]))\n" + _V_TAIL, "correct", None),
    ("average", "def average(runs):\n    return np.mean(runs, axis=0)\n", "correct", None),
    ("average", _V.replace("total / len(runs)", "total / len(runs) + 0.1"), "wrong", None),
    ("average", _V.replace("total / len(runs)", "total / 3"), "almost", None),
]
