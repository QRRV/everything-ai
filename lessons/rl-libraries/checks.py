# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-libraries: an RL library holds tested versions of the algorithms behind one interface. With
Stable-Baselines3, PPO learns CartPole in a few lines, and a fair result trains several seeds and tests each one on
many episodes. The course's runs of PPO, play_test and judge, the functions the learner writes, the stand-in models of
the check, the pictures, and the demo.

PPO is the algorithm of Schulman and colleagues (2017), here as Stable-Baselines3 2.9.2 has it (Raffin and colleagues,
2021, MIT), with its default settings; CartPole-v1 is an environment of Gymnasium 1.4.0 (Farama Foundation, MIT). The
code, the runs and the pictures are the course's own."""
import contextlib
import html
import math
import re
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from everything_ai import chart, live, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept

LESSON = "rl-libraries"
LANGS = ("en", "nl")
ENV = "CartPole-v1"
ROLLOUT = 2048        # the steps PPO plays before it learns from them: n_steps, the default of Stable-Baselines3
STEPS = (5000, 10000, 25000)   # what the demo offers to learn for
SEED_SETS = (0, 5)    # the demo trains seeds 0 to 4 or 5 to 9
SEEDS = 5             # seeds per set
TEST = 100            # test episodes per trained agent, seeds TEST_SEED to TEST_SEED + 99
TEST_SEED = 1000
WINDOW = 10           # the demo's chart: the mean return of the last this many episodes of training
GRID = 15             # the picture of the policy: a grid of this many angles by this many angular velocities
ANGLE = 0.2           # its angles, from -ANGLE to ANGLE radians: just inside 12°, where an episode ends
SPIN = 2.0            # its angular velocities, from -SPIN to SPIN


@contextlib.contextmanager
def one_thread():
    """PyTorch on one thread while it is open: as fast for PPO's small networks, and a run that does not depend on
    the number of cores (with four threads, the same seed gives other numbers)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def actual(steps: int) -> int:
    """The steps a run really takes: PPO learns from whole batches of ROLLOUT steps, so `learn` stops at the first
    whole batch at or past `steps`: 10,240 for 10,000."""
    return -(-int(steps) // ROLLOUT) * ROLLOUT


def seed_list(first: int) -> list:
    return list(range(first, first + SEEDS))


# --- the functions of the exercises, as the course writes them ----------------------------------------------------------

def play_test(model, seed):
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=seed)
    total = 0.0
    terminated, truncated = False, False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
    return total


def judge(returns):
    per_seed = returns.mean(axis=1)
    return np.median(per_seed), per_seed.min()


# --- the course's runs -------------------------------------------------------------------------------------------------------

def right_map(model) -> np.ndarray:
    """What the policy network says now, without changing it: the chance of pushing right for GRID angles (columns,
    left to right) by GRID angular velocities (rows, from fast to the right at the top to fast to the left at the
    bottom), with the cart in the middle and standing still."""
    angles = np.linspace(-ANGLE, ANGLE, GRID)
    spins = np.linspace(SPIN, -SPIN, GRID)
    obs = np.array([[0.0, 0.0, a, w] for w in spins for a in angles], dtype=np.float32)
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(obs)
        probs = model.policy.get_distribution(tensor).distribution.probs[:, 1]
    return probs.numpy().astype(float).reshape(GRID, GRID)


class _Follow(BaseCallback):
    """Hears the run of PPO: the return of every episode that ends, the policy after every batch, and the time."""

    def __init__(self, run: dict, show, limit: float | None):
        super().__init__()
        self.run, self.show, self.limit = run, show, limit
        self.begin = time.perf_counter()
        self.slow = False

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            episode = info.get("episode")
            if episode is not None:
                self.run["episodes"].append((int(self.num_timesteps), float(episode["r"])))
                if self.show is not None:
                    self.show(int(self.num_timesteps), self.run)
        if self.limit is not None and time.perf_counter() - self.begin > self.limit:
            self.slow = True
            return False
        return True

    def _on_rollout_end(self) -> None:
        self.run["maps"][int(self.num_timesteps)] = right_map(self.model)


def train(seed: int, steps: int, show=None, limit: float | None = None) -> dict | str:
    """One run of PPO on CartPole with the defaults of Stable-Baselines3, as the worked example makes it: the model,
    the return of every episode of training with the step it ended at, the policy as right_map at the start and after
    every batch, and the seconds it took. `show(step, run)` hears every episode that ends; "slow" after `limit`
    seconds."""
    env = gym.make(ENV)
    with one_thread():
        model = PPO("MlpPolicy", env, seed=seed, device="cpu")
        run = {"seed": seed, "steps": actual(steps), "episodes": [], "maps": {0: right_map(model)}, "model": model}
        follow = _Follow(run, show, limit)
        begin = time.perf_counter()
        model.learn(total_timesteps=steps, callback=follow)
        run["seconds"] = time.perf_counter() - begin
    env.close()
    return "slow" if follow.slow else run


def course_run(seed: int, steps: int) -> dict:
    return kept((LESSON, "run", seed, steps), lambda: train(seed, steps))


def test_returns(model, play=play_test) -> np.ndarray:
    """The returns of TEST test episodes of a trained model, each played by `play(model, seed)` with seeds TEST_SEED,
    TEST_SEED + 1, and so on."""
    with one_thread():
        return np.array([float(play(model, TEST_SEED + k)) for k in range(TEST)])


def course_returns(first: int, steps: int) -> np.ndarray:
    """The test returns of the course's seeds `first` to `first + 4` after `steps` steps: a row per seed, a column per
    test episode."""
    return kept((LESSON, "returns", first, steps),
                lambda: np.array([test_returns(course_run(s, steps)["model"]) for s in seed_list(first)]))


def curve(run: dict) -> list:
    """The demo's chart of a run: after every episode, the mean return of the last WINDOW episodes, at its step."""
    out = []
    returns = [r for _, r in run["episodes"]]
    for k, (t, _) in enumerate(run["episodes"]):
        part = returns[max(0, k + 1 - WINDOW):k + 1]
        out.append((t, sum(part) / len(part)))
    return out


def demo_test(runs: list, play, judge_, limit: float = 120.0) -> dict | str:
    """The demo after training: every trained model tested with the learner's play_test, the returns as a row per seed,
    and the learner's judge of them. "error" when either function stops with an error or gives something other than a
    number, "slow" after `limit` seconds."""
    from everything_ai.check import with_time_limit
    try:
        with contextlib.redirect_stdout(None):
            returns = with_time_limit(lambda: np.array([test_returns(r["model"], play) for r in runs]), limit)
            verdict = judge_(np.array(returns))
        median, worst = (float(v) for v in verdict)
    except Timeout:
        return "slow"
    except Exception:   # noqa: BLE001   the demo says what went wrong
        return "error"
    if not (math.isfinite(median) and math.isfinite(worst)):
        return "error"
    return {"returns": returns, "median": median, "worst": worst}


# --- words and numbers ---------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"


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


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    if out.startswith("-") and float(out.replace(",", "")) == 0:
        out = out[1:]
    out = out.replace("-", "−")
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def _join(values: list, lang: str) -> str:
    return ", ".join(values[:-1]) + (" en " if lang == "nl" else " and ") + values[-1]


# --- drawing ------------------------------------------------------------------------------------------------------------

_HALO = "paint-order:stroke;stroke:var(--ea-page);stroke-width:3px;stroke-linejoin:round"
TOP = 38   # the room above a picture for its title: a name, and a line under it


def _title(x: float, y: float, words: str) -> str:
    """A title of one or two lines: the part before ": " in bold, the rest in gray under it."""
    name, _, rest = words.partition(": ")
    out = f'<text x="{x}" y="{y}" style="font-size:13.5px;font-weight:600;fill:var(--ea-ink)">{html.escape(name)}</text>'
    if rest:
        out += f'<text x="{x}" y="{y + 16}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(rest)}</text>'
    return out


def _words(x: float, y: float, text_: str, anchor: str = "middle", size: float = 11.5, fill: str = "var(--ea-ink-2)",
           weight: int = 400) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" style="font-size:{size}px;font-weight:{weight};fill:{fill};'
            f'text-anchor:{anchor};{_HALO}">{html.escape(text_)}</text>')


def _svg(width: float, height: float, parts: list) -> str:
    return (f'<svg viewBox="0 0 {width:.0f} {height:.0f}" aria-hidden="true" style="flex:0 1 auto;width:100%;'
            f'max-width:{width:.0f}px;height:auto;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')


def _figure(svgs: list, alt: str, width: int) -> str:
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{width}px);display:flex;flex-wrap:wrap;gap:0.6rem 1.4rem;align-items:flex-start">'
            f'{"".join(svgs)}</figure>')


def _arrow(x0: float, y0: float, x1: float, y1: float, color: str = "var(--ea-ink-2)", width: float = 1.6) -> str:
    """A straight arrow from (x0, y0) to (x1, y1)."""
    d = math.hypot(x1 - x0, y1 - y0) or 1
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    bx, by = x1 - 7 * ux, y1 - 7 * uy
    return (f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{bx:.1f}" y2="{by:.1f}" style="stroke:{color};stroke-width:{width}"/>'
            f'<polygon points="{x1:.1f},{y1:.1f} {bx - 4 * uy:.1f},{by + 4 * ux:.1f} {bx + 4 * uy:.1f},{by - 4 * ux:.1f}" '
            f'style="fill:{color}"/>')


def _box(x: float, y: float, w: float, h: float, lines: list, learned: bool = False, size: float = 11) -> str:
    """A box with a line or two of words in it; one with numbers to learn is drawn in the color of what a model learns."""
    stroke = "var(--ea-param)" if learned else "var(--ea-line)"
    out = (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
           f'style="fill:var(--ea-page);stroke:{stroke};stroke-width:{1.6 if learned else 1.1}"/>')
    first = y + h / 2 + 4 - 7 * (len(lines) - 1)
    for k, line in enumerate(lines):
        out += _words(x + w / 2, first + 14 * k, line, size=size, fill="var(--ea-ink)" if k == 0 else "var(--ea-ink-2)")
    return out


_IDEA_TEXT = {
    "en": {"nets": "Inside PPO(\"MlpPolicy\", env): two networks",
           "policy": "policy network", "value": "value network", "in": ["4 numbers", "observation"],
           "hidden": ["64", "neurons"], "out_p": ["2 chances", "left, right"], "out_v": ["1 number", "V"],
           "count": "9,155 numbers to learn; 4,610 in the policy network, 4,545 in the value network",
           "loop": "PPO: play, judge, nudge",
           "play": ["1. play 2,048 steps", "with the policy as it is"],
           "judge": ["2. judge every action", "better or worse than V said?"],
           "nudge": ["3. nudge the policy", "10 passes, never far"],
           "again": "again, until the steps are done",
           "alt": "Two pictures. Left: the two networks inside PPO with MlpPolicy, each from the 4 numbers of an observation through two hidden layers of 64 neurons: the policy network to two chances, of pushing left and of pushing right, and the value network to one number, V. Right: the loop of PPO: play 2,048 steps with the policy as it is, judge every action against what V said, nudge the policy in 10 passes without going far, and again."},
    "nl": {"nets": "In PPO(\"MlpPolicy\", env): twee netwerken",
           "policy": "policy-netwerk", "value": "value-netwerk", "in": ["4 getallen", "observation"],
           "hidden": ["64", "neuronen"], "out_p": ["2 kansen", "links, rechts"], "out_v": ["1 getal", "V"],
           "count": "9.155 getallen om te leren; 4.610 in het policy-netwerk, 4.545 in het value-netwerk",
           "loop": "PPO: spelen, beoordelen, bijsturen",
           "play": ["1. speel 2.048 stappen", "met de huidige policy"],
           "judge": ["2. beoordeel elke action", "beter of slechter dan V zei?"],
           "nudge": ["3. stuur de policy bij", "10 rondes, nooit ver"],
           "again": "opnieuw, tot de stappen op zijn",
           "alt": "Twee afbeeldingen. Links: de twee netwerken in PPO met MlpPolicy, elk van de 4 getallen van een observation door twee hidden layers van 64 neuronen: het policy-netwerk naar twee kansen, op naar links en op naar rechts duwen, en het value-netwerk naar één getal, V. Rechts: de loop van PPO: speel 2.048 stappen met de huidige policy, beoordeel elke action tegen wat V zei, stuur de policy in 10 rondes bij zonder ver te gaan, en opnieuw."},
}


def _nets_svg(lang: str, W: int = 330) -> str:
    w = _IDEA_TEXT[lang]
    bw, bh, gap = 68, 40, 12
    parts = [_title(2, 14, w["nets"])]
    for k, (name, out, learned_out) in enumerate((("policy", "out_p", True), ("value", "out_v", True))):
        y = TOP + 22 + k * 92
        parts.append(_words(2, y - 7, w[name], anchor="start", size=11.5, fill="var(--ea-ink)", weight=600))
        xs = [2 + i * (bw + gap) for i in range(4)]
        parts.append(_box(xs[0], y, bw, bh, w["in"], size=10.5))
        parts.append(_box(xs[1], y, bw, bh, w["hidden"], learned=True, size=10.5))
        parts.append(_box(xs[2], y, bw, bh, w["hidden"], learned=True, size=10.5))
        parts.append(_box(xs[3], y, bw, bh, w[out], learned=learned_out, size=10.5))
        for i in range(3):
            parts.append(_arrow(xs[i] + bw + 1, y + bh / 2, xs[i + 1] - 1, y + bh / 2))
    for k, line in enumerate(w["count"].split("; ")):   # two lines
        parts.append(_words(2, TOP + 22 + 92 + bh + 24 + 15 * k, line, anchor="start", size=10.5))
    return _svg(W, TOP + 22 + 92 + bh + 49, parts)


def _loop_svg(lang: str, W: int = 330) -> str:
    w = _IDEA_TEXT[lang]
    bw, bh = 200, 40
    x = 40
    ys = [TOP + 8 + k * 64 for k in range(3)]
    parts = [_title(2, 14, w["loop"])]
    for y, key in zip(ys, ("play", "judge", "nudge")):
        parts.append(_box(x, y, bw, bh, w[key], learned=key == "nudge"))
    for k in range(2):
        parts.append(_arrow(x + bw / 2, ys[k] + bh + 1, x + bw / 2, ys[k + 1] - 1))
    right = x + bw + 30   # the way back: from the last box, along the right, into the first
    parts.append(f'<path d="M{x + bw + 1},{ys[2] + bh / 2} H{right} V{ys[0] + bh / 2} H{x + bw + 8}" '
                 f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.6"/>')
    parts.append(_arrow(x + bw + 9, ys[0] + bh / 2, x + bw + 1, ys[0] + bh / 2))
    parts.append(_words(x, ys[2] + bh + 22, w["again"], anchor="start", size=10.5))
    return _svg(W, ys[2] + bh + 32, parts)


def ppo_picture(lang: str) -> str:
    """The idea: what PPO("MlpPolicy", env) holds, and how it learns."""
    lang = lang if lang in LANGS else "en"
    return _figure([_nets_svg(lang), _loop_svg(lang)], _IDEA_TEXT[lang]["alt"], 690)


_MAP_TEXT = {
    "en": {"angle": "angle (degrees)", "spin": "angular velocity",
           "key": "darker: more often right, lighter: more often left; line: where 3 × angle + angular velocity is 0",
           "start": "before training", "after": "after {n} steps"},
    "nl": {"angle": "hoek (graden)", "spin": "hoeksnelheid",
           "key": "donkerder: vaker rechts, lichter: vaker links; lijn: waar 3 × hoek + hoeksnelheid 0 is",
           "start": "voor de training", "after": "na {n} stappen"},
}


def map_svg(probs, title: str, lang: str, W: int = 330) -> str:
    """The policy as a picture: for every angle (along the bottom) and angular velocity (up the side), with the cart
    in the middle and standing still, the chance that the network pushes right, as the strength of the color of what a
    model learns. The line is the rule of the lesson before, 3 × angle + angular velocity = 0."""
    w = _MAP_TEXT[lang]
    side = 228
    cell = side / GRID
    left, top = 58, TOP + 6
    parts = [_title(2, 14, title),
             f'<rect x="{left}" y="{top}" width="{side}" height="{side}" style="fill:var(--ea-page);stroke:var(--ea-line);stroke-width:1"/>']
    for i in range(GRID):
        for j in range(GRID):
            p = float(probs[i][j])
            if p > 0.004:
                parts.append(f'<rect x="{left + j * cell:.2f}" y="{top + i * cell:.2f}" width="{cell:.2f}" height="{cell:.2f}" '
                             f'shape-rendering="crispEdges" style="fill:var(--ea-param);fill-opacity:{p:.2f}"/>')

    def at(a: float, s: float) -> tuple:
        return left + (a + ANGLE) / (2 * ANGLE) * side, top + (SPIN - s) / (2 * SPIN) * side
    (x0, y0), (x1, y1) = at(-ANGLE, 3 * ANGLE), at(ANGLE, -3 * ANGLE)
    parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" style="stroke:var(--ea-ink);stroke-width:1.6"/>')
    for v in (-10, 0, 10):
        x = at(math.radians(v), 0)[0]
        parts.append(_words(x, top + side + 14, number(v, 0, lang), size=10.5))
    for v in (-SPIN, 0, SPIN):
        parts.append(_words(left - 5, at(0, v)[1] + 4, number(v, 0, lang), anchor="end", size=10.5))
    parts.append(_words(left + side / 2, top + side + 30, w["angle"], size=11))
    parts.append(_words(left - 30, top + side / 2, w["spin"], size=11).replace(
        "<text ", f'<text transform="rotate(-90 {left - 30} {top + side / 2})" ', 1))
    for k, line in enumerate(w["key"].split("; ")):
        parts.append(_words(2, top + side + 50 + 16 * k, line, anchor="start", size=10.5))
    return _svg(W, top + side + 74, parts)


def _when(n: int, lang: str) -> str:
    w = _MAP_TEXT[lang]
    return w["after"].format(n=number(n, 0, lang)) if n else w["start"]


# --- the stand-in models of the check -----------------------------------------------------------------------------------------

def _steady(obs) -> int:
    """Where the pole leans and where it falls, together: the line of the lesson before that keeps it up."""
    if 3 * obs[2] + obs[3] > 0:
        return 1
    return 0


def _falling(obs) -> int:
    """Where the pole falls: up until the cart runs off the track."""
    if obs[3] > 0:
        return 1
    return 0


def _leaning(obs) -> int:
    """Where the pole leans: every swing bigger than the one before."""
    if obs[2] > 0:
        return 1
    return 0


SWAP = 0.25   # without deterministic, a stand-in pushes the other way this often


class StandIn:
    """A stand-in for a trained model of Stable-Baselines3, with `predict` as the library has it, and a fixed policy
    inside instead of a network. With deterministic True it gives the policy's action; without, it draws, as PPO does
    in training: the policy's action 3 times in 4, the other one once. Its draws come from a generator of its own,
    seeded 0, and the check gives every call its own copy, so the same code gives the same returns."""

    def __init__(self, name: str, policy):
        self.name, self.policy = name, policy
        self.rng = np.random.default_rng(0)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        action = self.policy(observation)
        if not deterministic and self.rng.random() < SWAP:
            action = 1 - action
        return np.array(action), state

    def __repr__(self) -> str:
        return self.name


STEADY = StandIn("steady", _steady)
FALLING = StandIn("falling", _falling)
LEANING = StandIn("leaning", _leaning)
STAND_INS = (STEADY, FALLING, LEANING)


class NamedArray(np.ndarray):
    """Test returns, a row per seed, that show their name in a check message, such as `returns_10k`, instead of all
    their numbers. What is made from them, such as a mean, is a plain array or number again."""

    def __new__(cls, values, name: str):
        obj = np.array(values, dtype=float).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        self.name = getattr(obj, "name", None)

    def __getitem__(self, key):
        return np.ndarray.__getitem__(self.view(np.ndarray), key)

    def __array_wrap__(self, array, context=None, return_scalar=False):
        out = np.asarray(array)
        return out[()] if out.ndim == 0 else out

    def __deepcopy__(self, memo):
        return NamedArray(np.array(self), self.name)

    def __reduce__(self):
        return (NamedArray, (np.array(self), self.name))

    def __repr__(self):
        return self.name if self.name else repr(np.array(self))


# The test returns of the course's seeds 0 to 4 after 10,000 steps (10,240: whole batches), 100 per seed, seed after
# seed, as course_returns(0, 10000) gives them; tests/test_rl_libraries.py trains them again and compares.
_RETURNS_10K = """
500 500 500 500 259 500 500 500 500 271 285 297 356 500 279 500 500 192 286 500 307 297 500 256 500
326 298 500 500 333 500 482 203 255 500 304 500 500 500 500 500 277 500 500 254 500 500 500 312 500
287 290 500 304 500 500 500 500 500 500 299 500 500 500 289 296 274 354 500 368 354 500 288 500 222
352 500 500 287 249 500 500 500 246 500 500 304 500 327 500 500 500 500 500 500 500 500 321 299 500
500 465 500 500 238 495 500 500 500 238 250 270 500 500 241 500 500 186 248 500 265 268 474 241 500
290 264 500 500 296 500 370 195 226 500 268 500 500 475 500 402 259 500 463 277 500 500 500 265 500
263 262 500 256 500 500 500 500 480 500 254 500 500 412 256 258 269 320 500 322 313 500 259 431 199
327 500 500 261 220 393 455 500 285 500 500 259 479 277 500 500 463 500 500 500 500 394 290 257 500
247 282 191 500 249 356 269 363 389 259 217 297 368 188 179 241 500 160 176 500 281 189 346 203 500
317 282 477 181 328 208 273 168 164 287 230 344 220 282 347 197 173 500 500 328 176 358 454 169 366
167 277 484 181 381 204 216 272 500 500 161 262 191 284 271 167 296 361 244 361 338 347 171 313 156
354 193 273 277 159 180 329 470 311 185 181 181 287 192 359 500 292 500 185 234 500 291 298 169 473
238 276 190 500 246 354 484 357 377 247 222 280 369 188 178 200 500 156 175 500 279 197 339 205 500
306 279 500 178 321 206 270 159 163 283 227 342 218 500 346 191 166 500 500 314 175 352 457 163 366
167 281 476 166 370 194 212 268 500 500 161 255 184 283 264 163 286 358 240 350 330 339 169 313 153
344 189 267 273 151 180 328 485 305 183 174 177 274 190 368 500 282 500 183 231 500 284 303 169 480
435 334 500 500 250 390 500 407 419 254 232 291 412 228 230 438 500 184 228 500 278 254 374 219 500
314 279 500 234 327 500 304 193 209 500 248 398 500 500 380 308 197 500 500 317 440 396 500 201 405
236 275 500 210 418 259 269 500 500 500 200 500 258 324 266 195 290 344 411 352 331 393 217 344 188
342 500 291 271 197 302 370 500 295 230 224 235 500 267 417 500 500 500 230 500 500 311 297 243 500
"""
RETURNS_10K = NamedArray(np.array(_RETURNS_10K.split(), dtype=float).reshape(SEEDS, TEST), "returns_10k")
RETURNS_25K = NamedArray(np.full((SEEDS, TEST), 500.0), "returns_25k")   # the same seeds after 25,000 steps: all 500
# 7 seeds of 20 test episodes each, made up: two that never learned, three on the way, two that hold the pole up
MIXED = NamedArray([[9, 11, 10, 12, 8, 10, 9, 11, 10, 10, 12, 9, 10, 11, 8, 10, 9, 10, 11, 10],
                    [500, 498, 500, 500, 467, 500, 500, 500, 500, 500, 500, 500, 500, 431, 500, 500, 500, 500, 500, 500],
                    [210, 250, 190, 230, 260, 240, 200, 220, 180, 270, 230, 210, 250, 240, 190, 220, 230, 260, 200, 220],
                    [120, 140, 500, 130, 110, 125, 135, 500, 115, 145, 130, 120, 140, 125, 135, 110, 150, 130, 120, 140],
                    [500] * 20,
                    [330, 310, 350, 320, 300, 340, 360, 330, 310, 320, 350, 340, 330, 300, 320, 310, 360, 340, 330, 320],
                    [21, 18, 25, 19, 23, 20, 22, 24, 17, 21, 20, 19, 23, 22, 18, 25, 21, 20, 19, 24]], "returns_mixed")
TWO = NamedArray([[100, 300, 500], [450, 470, 490]], "returns_two")   # two seeds of three test episodes

PLAY_CASES = [(STEADY, 0), (STEADY, 1), (FALLING, 0), (FALLING, 3), (LEANING, 2)]
JUDGE_CASES = [(RETURNS_10K,), (RETURNS_25K,), (MIXED,), (TWO,)]


def _fine(x) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(x)


def same_number(a, b) -> bool:
    return _fine(a) and _fine(b) and abs(float(a) - float(b)) < 1e-9


def same_pair(a, b) -> bool:
    """judge: two numbers, the median and the worst, each as the course's."""
    return (isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == len(b) == 2
            and all(_fine(x) and abs(float(x) - float(y)) < 1e-6 for x, y in zip(a, b)))


def _short(v) -> str:
    if _fine(v):
        f = float(v)
        return str(int(f)) if f.is_integer() else f"{round(f, 4)!r}"
    text_ = repr(v)
    return text_ if len(text_) <= 80 else text_[:77] + "..."


def show_value(v) -> dict:
    shown = f"`{_short(v)}`"
    return {"en": shown, "nl": shown}


def show_pair(v) -> dict:
    if isinstance(v, (tuple, list)) and len(v) == 2 and all(_fine(x) for x in v):
        shown = f"`({_short(v[0])}, {_short(v[1])})`"
    elif isinstance(v, np.ndarray):
        shown = f"an array of shape `{v.shape}`"
        return {"en": shown, "nl": f"een array met shape `{v.shape}`"}
    else:
        shown = f"`{_short(v)}`"
    return {"en": shown, "nl": shown}


@dataclass
class PlayBox(FunctionBox):
    """play_test in a code box: the check calls it with a stand-in model and a seed, each with its own copy of the
    stand-in, so its draws start the same every time."""

    def _call_text(self, case) -> str:
        return f"play_test({case[0].name}, {case[1]})"


# --- the pictures of the learner's work -----------------------------------------------------------------------------------

_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 4   # what the learner's function gets for all the calls of one picture together


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make_, seconds: float = PICTURE_SECONDS):
    """make_() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    import io
    from everything_ai.check import with_time_limit
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make_, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


_YOURS_TEXT = {
    "en": {"play_test": {"title": "Your tests of the stand-ins",
                         "intro": "The return of one test episode, played by your `play_test` with each stand-in and the seeds 0 to 4; in the color of mistakes, a return that differs from the course's `play_test`.",
                         "seed": "seed {s}", "say": "With the most probable action every time, `steady` keeps the pole up for all 500 steps in every episode, `falling` until the cart runs off the track, and `leaning` until the pole falls.",
                         "none": "Your `play_test` stops with an error, takes too long, or gives something other than a number, so the stand-ins cannot be tested yet."},
           "judge": {"title": "Your judgment of the course's seeds",
                     "intro": "The mean of the 100 test episodes of each of the course's seeds after 10,000 steps, `returns_10k`, with your median and your worst seed as lines; in the color of mistakes, a line that differs from the course's `judge`.",
                     "panel": "The mean of 100 test episodes, per seed", "x": "steps the pole stays up", "seed": "seed {s}",
                     "median": "your median", "worst": "your worst",
                     "say": "Your `judge` gives a median of {median} and a worst seed of {worst}; the best seed reaches {best}.",
                     "none": "Your `judge` stops with an error, takes too long, or gives something other than two numbers, so nothing can be drawn yet."}},
    "nl": {"play_test": {"title": "Je tests van de stand-ins",
                         "intro": "De return van één testepisode, gespeeld door je `play_test` met elke stand-in en de seeds 0 tot en met 4; in de kleur van fouten een return die anders is dan met de `play_test` van de course.",
                         "seed": "seed {s}", "say": "Met elke keer de waarschijnlijkste action houdt `steady` de paal in elke episode alle 500 stappen overeind, `falling` tot het karretje van de baan rijdt, en `leaning` tot de paal valt.",
                         "none": "Je `play_test` stopt met een fout, duurt te lang, of geeft iets anders dan een getal, dus kunnen de stand-ins nog niet getest worden."},
           "judge": {"title": "Je oordeel over de seeds van de course",
                     "intro": "Het gemiddelde van de 100 testepisodes van elke seed van de course na 10.000 stappen, `returns_10k`, met je mediaan en je slechtste seed als lijnen; in de kleur van fouten een lijn die anders is dan met de `judge` van de course.",
                     "panel": "Het gemiddelde van 100 testepisodes, per seed", "x": "stappen dat de paal blijft staan", "seed": "seed {s}",
                     "median": "je mediaan", "worst": "je slechtste",
                     "say": "Je `judge` geeft een mediaan van {median} en een slechtste seed van {worst}; de beste seed haalt {best}.",
                     "none": "Je `judge` stopt met een fout, duurt te lang, of geeft iets anders dan twee getallen, dus is er nog geen afbeelding."}},
}


def play_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["play_test"]
    fn = _fn(seen)
    import copy
    got = _call(lambda: [[fn(copy.deepcopy(m), s) for s in range(5)] for m in STAND_INS]) if fn is not None else None
    if not isinstance(got, list) or not all(_fine(v) for row in got for v in row):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    left, col, row, top = 80, 64, 28, 30
    W = left + 5 * col + 4
    parts = []
    for s in range(5):
        parts.append(_words(left + (s + 0.5) * col, top - 10, w["seed"].format(s=s), size=11))
    for i, m in enumerate(STAND_INS):
        y = top + i * row
        parts.append(f'<rect x="0" y="{y}" width="{W}" height="{row}" style="fill:{"var(--ea-inset)" if i % 2 == 0 else "var(--ea-page)"}"/>')
        parts.append(f'<text x="6" y="{y + 18}" style="font-family:var(--ea-font-code);font-size:12px;fill:var(--ea-ink)">{html.escape(m.name)}</text>')
        for s in range(5):
            mine = got[i][s]
            want = play_test(copy.deepcopy(m), s)
            wrong = not same_number(mine, want)
            parts.append(_words(left + (s + 0.5) * col, y + 18, number(float(mine), 0, lang), size=12,
                                fill="var(--ea-loss-text)" if wrong else "var(--ea-ink)", weight=600 if wrong else 400))
    svg = _figure([_svg(W, top + 3 * row + 4, parts)], w["title"], W)
    return yours.figure(w["title"], _note(w["intro"]) + svg + _note(w["say"]), alt=w["title"])


def judge_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["judge"]
    fn = _fn(seen)
    import copy
    got = _call(lambda: fn(copy.deepcopy(RETURNS_10K))) if fn is not None else None
    if not (isinstance(got, (tuple, list)) and len(got) == 2 and all(_fine(v) for v in got)):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    median, worst = (float(v) for v in got)
    want_median, want_worst = (float(v) for v in judge(np.array(RETURNS_10K)))
    means = np.array(RETURNS_10K).mean(axis=1)
    p = chart.Panel((150, 500), (0.3, SEEDS + 0.9), title=w["panel"], xlabel=w["x"], ylabel="", xticks=[200, 300, 400, 500],
                    yticks=[])
    # the worst seed's name to the left of its line, the median's to the right: the worst is never above the median
    for value, role, words, k in ((median, "ink", w["median"], 0), (worst, "ink-2", w["worst"], 1)):
        wrong = abs(value - (want_median, want_worst)[k]) > 1e-6
        shown = min(max(value, 150), 500)
        p.line([shown, shown], [0.3, SEEDS + 0.5], "loss" if wrong else role, 1.6, dashed=k == 1)
        if k == 0:
            anchor = "start" if shown <= 420 else "end"
        else:
            anchor = "end" if shown >= 240 else "start"
        p.text(shown + (5 if anchor == "start" else -5), SEEDS + 0.7, words, "loss" if wrong else role, anchor=anchor, halo=True)
    p.points(list(means), [s + 1 for s in range(SEEDS)], "param", r=4)
    for s, m in enumerate(means):
        p.text(m + 6, s + 1, w["seed"].format(s=s), "ink-2", anchor="start", halo=True)
    svg = chart.figure([p], alt=w["panel"], width=520, height=250)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    say = w["say"].format(median=number(median, 2, lang), worst=number(worst, 2, lang), best=number(float(means.max()), 2, lang))
    return yours.figure(w["title"], _note(w["intro"]) + svg + _note(say), alt=w["title"])


# --- the demo ----------------------------------------------------------------------------------------------------------------

DEMO_TEXT = {
    "en": {"title": "Seed {s}: return per episode", "x": "steps", "y": "mean of the last {w}",
           "alt": "The return of every episode of training of seed {s}, and the mean of the last {w}, against the steps so far.",
           "steps": "Steps", "seeds": "Seeds", "set": "{a} to {b}", "map": "The policy of seed {s}: {when}",
           "learning": "**Seed {s}, {k} of {n}: learning** …", "testing": "**Testing** the {n} trained agents, 100 episodes each …",
           "table": "| Seed | mean of 100 test episodes | episodes that reach 500 | seconds of training |",
           "verdict": "With {who}, your `judge` gives a median of **{median}** and a worst seed of **{worst}** over the {n} seeds, after {steps} steps each.",
           "say": "Seed {s} after {n} steps: the mean return of the last {w} episodes of training is {m}."},
    "nl": {"title": "Seed {s}: return per episode", "x": "stappen", "y": "gemiddelde van de laatste {w}",
           "alt": "De return van elke episode van de training van seed {s}, en het gemiddelde van de laatste {w}, tegen de stappen tot nu toe.",
           "steps": "Stappen", "seeds": "Seeds", "set": "{a} tot en met {b}", "map": "De policy van seed {s}: {when}",
           "learning": "**Seed {s}, {k} van {n}: bezig met leren** …", "testing": "**Bezig met testen** van de {n} getrainde agents, 100 episodes elk …",
           "table": "| Seed | gemiddelde van 100 testepisodes | episodes die 500 halen | seconden training |",
           "verdict": "Met {who} geeft je `judge` een mediaan van **{median}** en een slechtste seed van **{worst}** over de {n} seeds, na {steps} stappen elk.",
           "say": "Seed {s} na {n} stappen: de gemiddelde return van de laatste {w} episodes van de training is {m}."},
}


def steps_options(lang: str) -> dict:
    return {number(s, 0, lang): s for s in STEPS}


def set_options(lang: str) -> dict:
    w = DEMO_TEXT[lang if lang in LANGS else "en"]
    return {w["set"].format(a=f, b=f + SEEDS - 1): f for f in SEED_SETS}


def chart_labels(seed: int, lang: str) -> dict:
    w = DEMO_TEXT[lang]
    return {"title": w["title"].format(s=seed), "x": w["x"], "y": w["y"].format(w=WINDOW),
            "alt": w["alt"].format(s=seed, w=WINDOW)}


def now_picture(run: dict, lang: str) -> str:
    """Beside the chart of the demo: what the policy network of the run does now."""
    lang = lang if lang in LANGS else "en"
    model = run.get("model")
    probs = right_map(model) if model is not None else run["maps"][0]
    n = int(model.num_timesteps) if model is not None else 0
    title = DEMO_TEXT[lang]["map"].format(s=run["seed"], when=_when(n, lang))
    return _figure([map_svg(probs, title, lang)], title, 330)


def table(runs: list, returns, lang: str) -> str:
    """The demo after the test: a row per seed, in Markdown."""
    rows = [DEMO_TEXT[lang]["table"], "|---|---|---|---|"]
    for run, r in zip(runs, returns):
        rows.append(f"| {run['seed']} | {number(float(np.mean(r)), 2, lang)} | {int(np.sum(np.array(r) >= 500))} | "
                    f"{number(run['seconds'], 1, lang)} |")
    return "\n".join(rows)


def _curve_panel(run: dict, n: int, lang: str) -> chart.Panel:
    w = DEMO_TEXT[lang]
    p = chart.Panel((0, run["steps"]), (0, 500), title=w["title"].format(s=run["seed"]), xlabel=w["x"],
                    ylabel=w["y"].format(w=WINDOW), xticks=ticks(run["steps"]))
    points = curve(run)
    if points:
        p.line([t for t, _ in run["episodes"]], [r for _, r in run["episodes"]], "axis", 0.8)
        p.line([t for t, _ in points], [m for _, m in points], "ink", 2.4)
    if 0 < n <= run["steps"]:
        p.line([n, n], [0, 500], "ink-2", 1.2)
    return p


def ticks(steps: int) -> list:
    """The x-axis of a run: round ticks that fit, without a last one on the right edge."""
    step = {6144: 2000, 10240: 2500, 26624: 5000}.get(steps, 5000)
    return list(range(0, steps - step // 2, step))


def looks(run: dict) -> list:
    """The stops of the rewind: the start, and after every batch."""
    return sorted(run["maps"])


def mean_at(run: dict, n: int) -> float | None:
    """The demo's chart at step n: the mean return of the last WINDOW episodes that ended by then."""
    done = [r for t, r in run["episodes"] if t <= n]
    return sum(done[-WINDOW:]) / len(done[-WINDOW:]) if done else None


def rewind_frame(run: dict, n: int, lang: str) -> tuple:
    """A stop of the rewind after the demo: the chart with a line at n, the policy as it was then, and a sentence."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    n = min(run["maps"], key=lambda k: abs(k - int(n)))
    svg = chart.figure([_curve_panel(run, n, lang)], alt=w["alt"].format(s=run["seed"], w=WINDOW), width=330, height=240)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    title = w["map"].format(s=run["seed"], when=_when(n, lang))
    picture = _figure([map_svg(run["maps"][n], title, lang)], title, 330)
    m = mean_at(run, n)
    say = "" if m is None else w["say"].format(s=run["seed"], n=number(n, 0, lang), w=WINDOW, m=number(m, 1, lang))
    return live.rewound(svg, [picture], width=330), say


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["play_test", "judge", "fair", "library"]


def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    import ast
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


# exercise 1: play_test
def _p(line: str = "model.predict(obs, deterministic=True)") -> str:
    return ("def play_test(model, seed):\n"
            "    env = gym.make(\"CartPole-v1\")\n"
            "    obs, info = env.reset(seed=seed)\n"
            "    total = 0.0\n"
            "    terminated, truncated = False, False\n"
            "    while not (terminated or truncated):\n"
            f"        action, _ = {line}\n"
            "        obs, reward, terminated, truncated, info = env.step(action)\n"
            "        total += reward\n"
            "    return total\n")


_P_START = _p("...")
_P = _p()
_P_DRAWS = _p("model.predict(obs)")
_P_PLACE = _p("model.predict(obs, True)")
_P_RANDOM = _p("env.action_space.sample(), None")
P_NAMES = {"gym": gym}


def _placed(got, text_) -> bool:
    """True as the second value of predict: the place of `state`, so predict still draws."""
    return re.search(r"predict\(\s*obs\s*,\s*True\s*\)", text_) is not None


def _sampled(got, text_) -> bool:
    return "action_space.sample" in text_


# exercise 2: judge
def _j(line: str = "returns.mean(axis=1)") -> str:
    return ("def judge(returns):\n"
            f"    per_seed = {line}\n"
            "    return np.median(per_seed), per_seed.min()\n")


_J_START = _j("...")
_J = _j()
_J_EPISODES = _j("returns.mean(axis=0)")
_J_ONE = _j("returns.mean()")
_J_BEST = _j("returns.max(axis=1)")
_J_HUNDRED = _j("returns.sum(axis=1) / 100")
J_NAMES = {"np": np}


EXERCISES = {
    "play_test": PlayBox(
        name="play_test",
        function="play_test",
        start=_P_START,
        solution=_P,
        names=P_NAMES,
        compare=same_number,
        show=show_value,
        visible=PLAY_CASES[:1],
        hidden=PLAY_CASES[1:],
        first=[_DOTS],
        picture=play_picture,
        mistakes=[
            Mistake("placed", matches=_placed,
                    en="`{call}` gives {got}, and it should give {want}: the second place of `predict` is for `state`, so your `True` goes there, and `predict` still draws its action as in training. How do you give `predict` the `True` for `deterministic` by its name?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de tweede plaats van `predict` is voor `state`, dus je `True` komt daar terecht, en `predict` trekt nog steeds een action zoals bij de training. Hoe geef je `predict` de `True` voor `deterministic` met de naam erbij?"),
            Mistake("sampled", matches=_sampled,
                    en="`{call}` gives {got}, and it should give {want}: your loop takes random actions from the action space, so the test says nothing about the model. Which method of `model` gives the action of the trained agent?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je loop neemt willekeurige actions uit de action space, dus zegt de test niets over het model. Welke method van `model` geeft de action van de getrainde agent?"),
            Mistake("draws", wrong=_P_DRAWS,
                    en="`{call}` gives {got}, and it should give {want}: with `deterministic` left out or `False`, `predict` draws its action from the chances of the network, as in training, so the stand-in pushes the other way now and then. How do you make `predict` take the most probable action every time?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: met `deterministic` weggelaten of `False` trekt `predict` een action uit de kansen van het netwerk, zoals bij de training, dus duwt de stand-in af en toe de andere kant op. Hoe laat je `predict` elke keer de waarschijnlijkste action nemen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a call of `model.predict` that gives the most probable action for `obs`.",
                      "nl": "Vervang de `...` zo dat `model.predict` de waarschijnlijkste action voor `obs` geeft."},
            "arguments": {"en": "The check calls `{call}`, with a model and a seed, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(model, seed)`?",
                          "nl": "De check roept `{call}` aan, met een model en een seed, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(model, seed)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your loop ask `model.predict` for the most probable action at every step, and add up every reward?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Vraagt je loop bij elke stap `model.predict` om de waarschijnlijkste action, en telt de loop elke reward op?"},
            "almost": {"en": "Right with `steady`. But `{call}` gives {got}, and it should give {want}. Does your loop take its actions from `model`, whatever the model is?",
                       "nl": "Goed met `steady`. Maar `{call}` geeft {got}, en het moet {want} zijn. Neemt je loop de actions van `model`, welk model het ook is?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return total`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return total`?"},
            "correct": {"en": "Right in all {n} cases: `steady` holds the pole for all 500 steps, `falling` until the cart runs off the track, and `leaning` far shorter. In the picture, which stand-in does best with every seed?",
                        "nl": "Goed in alle {n} gevallen: `steady` houdt de paal alle 500 stappen overeind, `falling` tot het karretje van de baan rijdt, en `leaning` veel korter. Welke stand-in doet het in de afbeelding met elke seed het best?"},
        },
    ),
    "judge": FunctionBox(
        name="judge",
        function="judge",
        start=_J_START,
        solution=_J,
        names=J_NAMES,
        compare=same_pair,
        show=show_pair,
        visible=JUDGE_CASES[:1],
        hidden=JUDGE_CASES[1:],
        first=[_DOTS],
        picture=judge_picture,
        mistakes=[
            Mistake("episodes", wrong=_J_EPISODES,
                    en="`{call}` gives {got}, and it should give {want}: `axis=0` takes the mean down each column, over the seeds, so you get one mean per test episode instead of one per seed. Along which axis lie the test episodes of one seed?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `axis=0` neemt het gemiddelde langs elke kolom, over de seeds, dus krijg je één gemiddelde per testepisode in plaats van één per seed. Langs welke as liggen de testepisodes van één seed?"),
            Mistake("one", wrong=_J_ONE,
                    en="`{call}` gives {got}, and it should give {want}: without an axis, `mean` takes one mean over every number of the array, so the median and the worst seed are that same number. How do you keep one mean for each seed?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: zonder as neemt `mean` één gemiddelde over alle getallen van de array, dus zijn de mediaan en de slechtste seed hetzelfde getal. Hoe houd je één gemiddelde per seed?"),
            Mistake("best", wrong=_J_BEST,
                    en="`{call}` gives {got}, and it should give {want}: you take the best test episode of each seed, so a seed that drops the pole in most of its episodes still counts as 500. What should each seed get, so that every one of its test episodes counts?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je neemt de beste testepisode van elke seed, dus telt een seed die de paal in de meeste episodes laat vallen nog steeds als 500. Wat moet elke seed krijgen, zodat elke testepisode van een seed meetelt?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the mean of the test returns of each seed: one number per row of `returns`.",
                      "nl": "Vervang de `...` door het gemiddelde van de testreturns van elke seed: één getal per rij van `returns`."},
            "arguments": {"en": "The check calls `{call}`, with one array of test returns, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(returns)`?",
                          "nl": "De check roept `{call}` aan, met één array van testreturns, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(returns)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `per_seed` hold the mean of each row of `returns`, one number per seed?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Bevat `per_seed` het gemiddelde van elke rij van `returns`, één getal per seed?"},
            "almost": {"en": "Right for `returns_10k`. But `{call}` gives {got}, and it should give {want}. Does your function work for any number of seeds and of test episodes?",
                       "nl": "Goed voor `returns_10k`. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor elk aantal seeds en testepisodes?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with the line that starts with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met de regel die met `return` begint?"},
            "correct": {"en": "Right in all {n} cases: after 10,000 steps the course's seeds have a median of 353.09 and a worst seed of 292.57, and after 25,000 steps both are 500. In the picture, how far apart are the best and the worst seed?",
                        "nl": "Goed in alle {n} gevallen: na 10.000 stappen hebben de seeds van de course een mediaan van 353,09 en een slechtste seed van 292,57, en na 25.000 stappen zijn beide 500. Hoe ver liggen de beste en de slechtste seed in de afbeelding uit elkaar?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("play_test", _P_START, "empty", None),
    ("play_test", _P, "correct", None),
    ("play_test", _p("model.predict(obs, deterministic=...)"), "wrong", "dots-left"),
    ("play_test", _P_DRAWS, "wrong", "draws"),
    ("play_test", _p("model.predict(obs, deterministic=False)"), "wrong", "draws"),
    ("play_test", _P_PLACE, "wrong", "placed"),
    ("play_test", _P_RANDOM, "wrong", "sampled"),
    ("play_test", _P.replace("    return total\n", ""), "wrong", "no-return"),
    # other right answers: predict's whole answer taken apart later, and the steps counted instead of the rewards
    ("play_test", _p("model.predict(obs, deterministic=True)[0], None"), "correct", None),
    ("play_test", _P.replace("        total += reward\n", "        total += 1\n"), "correct", None),
    # the stand-in steady's own policy, whatever the model is: right with steady only
    ("play_test", _p("int(3 * obs[2] + obs[3] > 0), None"), "almost", None),
    ("judge", _J_START, "empty", None),
    ("judge", _J, "correct", None),
    ("judge", _j("returns.mean(axis=...)"), "wrong", "dots-left"),
    ("judge", _J_EPISODES, "wrong", "episodes"),
    ("judge", _J_ONE, "wrong", "one"),
    ("judge", _J_BEST, "wrong", "best"),
    ("judge", _J.replace("    return np.median(per_seed), per_seed.min()\n", "    np.median(per_seed), per_seed.min()\n"), "wrong", "no-return"),
    # other right answers: the last axis, and NumPy's function instead of the method
    ("judge", _j("returns.mean(axis=-1)"), "correct", None),
    ("judge", _j("np.mean(returns, axis=1)"), "correct", None),
    # divided by the 100 test episodes of the example: right for 100 episodes only
    ("judge", _J_HUNDRED, "almost", None),
]
