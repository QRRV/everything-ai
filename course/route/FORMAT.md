# Route data format

This folder describes the course as data: which lessons exist, in which order a learner meets them, and where every topic of the map lives. Nothing here is a lesson yet; it is the plan the lessons will be built from. `tools/check_route.py` checks all of it and writes `rapport/`.

There are three layers:

1. **The map** (`onderzoek/kaart/kaart_v2.yaml`, owned by the map thread) says *what exists*: 27 domains, 623 topics, about 3,400 subtopics, each with a level N1 to N5 and prerequisite topics.
2. **Lessons** (`lessen/*.yaml`) say *how it is taught*: one idea per lesson, with what it needs and what it introduces.
3. **Routes** (`routes.yaml`) say *in which order*: ordered lists of lesson ids. A lesson can be in many routes.

Branches (`takken.yaml`) tie the map to the grid of the course: every map topic has exactly one home branch.

## Lesson files: `lessen/*.yaml`

```yaml
tak: stam          # default branch for the lessons in this file (a lesson may override it)
map: v2.2          # map version that the "covers" references point into
lessons:
- id: py-first-program
  title: "Your first program"
  level: basis
  module: python
  idea: "A program is a list of instructions that runs from top to bottom; print shows a result."
  covers: [python#0]
  requires: [start-how-this-course-works]
  introduces: [c:program, py:print, py:string-literal]
  uses: [q:run-cell]
  ladder: [python]
  needs: any
```

| field | required | meaning |
|---|---|---|
| `id` | yes | Stable id in English kebab-case, never a number. Never renamed once lessons exist; the app stores progress by it. Prefix shows the subject (`py-`, `np-`, `ml-`, `nn-`, `torch-`, `cnn-`, `attn-`, `lm-`, `tok-`, `gpt-`, `llm-`, `sys-`, `res-`, `repro-`, `math-`, `prob-`, `stats-`, `info-`, `opt-`, `data-`, `rai-`, `rl-`, `search-`, `start-`). |
| `tak` | yes (or from the file) | Home branch, one of the ids in `takken.yaml`. |
| `title` | yes | English title. The Dutch title is written with the lesson. |
| `level` | yes | `basis`, `gevorderd` or `expert`. |
| `module` | yes | Group of lessons inside a branch and level, used by the app to show chapters. |
| `idea` | yes | The one idea of the lesson, in one or two sentences. If it needs "and also", it is two lessons. |
| `covers` | yes | Map subtopics this lesson teaches, as `topic-id#index` into the map version in `map:`. `[]` means no subtopic fits, on purpose. |
| `requires` | yes | Lesson ids that must come before. Only the first lesson has none. |
| `introduces` | no | Inventory items this lesson explains for the first time. Each item is introduced by exactly one lesson. Symbols (`s:`) are introduced where the notation first appears, even when the termen list already defines them. |
| `uses` | no | Items this lesson uses without explaining them. Each must be introduced somewhere in its `requires` chain. Filled in fully when the lesson is built. |
| `mentions` | no | Items the lesson names in passing without relying on them, such as "your first language model" in the letter-count project. The name check skips them. |
| `warns` | no | Words that mean something else in daily life or in another field (`token`, `kernel`, `bias`, `class`); the lesson says so. The name check treats them as homonyms. |
| `ladder` | no | Code rungs used: `python` (by hand), `numpy`, `torch`. |
| `kind` | no | `project` for a step of a running project; `refresher` for a short repair lesson on something taught earlier; `review` for mixed practice of earlier lessons, including a standalone mini-project. Refresher and review lessons introduce nothing new. |
| `project` | with kind project | Running project id from `takken.yaml` (`nn`, `cnn`, `llm`, `repro`, ...). |
| `data` | no | Dataset, model or code ids from `bronnen.yaml`. `[]` says on purpose that the lesson uses no external data. |
| `reference_data` | see rule 15 | The data the course trains its own reference checkpoint for this step on, when that differs from `data`. `[]` means the course hands out no checkpoint for this step. |
| `minutes` | no | Study time in minutes, without waiting for training. Filled in when the lesson is built; `didactiek/generieke-principes.md` aims at 10 to 15. |
| `compute_minutes` | see rule 14 | Training or compute time on the recommended spec (RTX 3060 12 GB), an estimate until measured. `paper` for the own-reproduction steps whose time depends on the chosen paper (see `papers.yaml`). |
| `fallback` | see rule 14 | What a learner without the hardware does: `cpu` (a smaller version runs on a CPU), `checkpoint` (load the course's trained result and go on), `recorded` (study measurements and logs from the reference machine). |
| `goal` | no | The goal as something the learner can do ("compute the MSE of a prediction"), for the lesson header and summary. |
| `hard` | no | `true` for a lesson that is tougher than its neighbors; the lesson says so upfront. |
| `look_back` | no | One earlier lesson (in the `requires` chain) that the lesson's look-back question comes from. (Not called `review`, to avoid a clash with `kind: review`.) |
| `needs` | yes | Hardware: `any` (runs on a CPU), `gpu8` (8 GB NVIDIA GPU), `rec` (recommended spec, RTX 3060 12 GB or better), `nvidia-linux` (needs Linux or WSL2 with an NVIDIA GPU, for Triton and FlashAttention). |

Inventory item prefixes: `c:` concept, `s:` symbol or notation, `py:` Python or library API, `v:` view (a kind of plot or picture), `q:` question form (a kind of exercise), `p:` procedure. An item is written in a lesson text as its name with spaces: `c:causal-mask` is "causal mask".

## Routes: `routes.yaml`

A route has an `id`, titles, `milestones` (shown when a learner passes a step) and `steps`. A step is a lesson id, or `{id: ..., extra: true}` for an optional deepening step (verdieping) in that route. Being extra is a property of the step in a route, not of the lesson: the same lesson can be core in one route and extra in another.

`goals` lists example goal routes. They are computed, not written: the lessons a goal lesson needs, in the order of the main route.

## Branches: `takken.yaml`

Each branch has an id, titles, the map domains it owns by default, a `status` (`main-route` or `later`), a proposed `build_order`, its running `project` (an id with an English `title` and a Dutch `title_nl`) and its final project. A later branch has `attach`: its attach points, the lessons of the main route it connects to, in route order. The branch can begin after the first one; its later lessons build on the others too. Like `build_order`, a proposal. `overrides` moves single topics to another branch than their domain's default. A main-route branch has `rest_after`: the branch after which the rest of it is built (what the main route does not cover and no earlier branch has built ahead).

## Papers and sources

`papers.yaml` lists the guided paper and the papers a learner can choose for the own reproduction, each with the hardware it needs and a compute estimate. `bronnen.yaml` lists every dataset, model and code id used in a `data` field, with its license, whether the course may host it, and `checkpoints: false` when the course may not hand out models trained on it.

## Rules the checker enforces

Errors (the route is not valid until these are gone):

1. Lesson ids are unique; every `requires` id exists; `requires` has no cycles.
2. A lesson's level is at least the level of every lesson it requires.
3. A lesson's level is at least the level of every subtopic it covers (basis = N1 and N2, gevorderd = N3, expert = N4 and N5).
4. Every `uses` item is introduced by the lesson itself or by a lesson in its `requires` chain. Because this uses the chain and not the route order, any route that respects `requires` is free of forward references.
5. Every item is introduced exactly once and has a known prefix; every `mentions` item exists.
6. Every `covers` reference exists in its map version.
7. A lesson on the `numpy` or `torch` rung builds on the lesson that introduces that rung.
8. Hardware fits the level: basis lessons run on a CPU, gevorderd lessons need at most an 8 GB GPU.
9. Every step of a route exists, appears once, comes after everything it requires, and the route contains everything its steps require.
10. A core step of a route never requires an extra step of that route.
11. The main route never goes down a level.
12. Every map topic has a home branch; overrides name existing topics and branches.
13. Every `data` and `reference_data` id is in `bronnen.yaml`; papers and branch finals point to existing lessons; goals are in the main route; every project has a `title` and a `title_nl`; every later branch has `attach` points, core steps of the main route in route order, and a main-route branch has none; every own-choice paper has `needs` and a compute estimate, and at least one runs on a CPU.
14. `kind`, `project`, `minutes`, `compute_minutes`, `fallback`, `hard`, `goal` and `look_back` have valid values; `kind: project` has a `project` and the other way round; a core step of the main route computes at most 15 minutes; a core step of the main route that needs a GPU has a `fallback` (unless its time depends on the chosen paper); `look_back` points into the `requires` chain.
15. A project step that trains on data with `checkpoints: false` has `reference_data` with data the course may train its handed-out checkpoints on, or `[]`.

Warnings:

- a lesson without a `covers` field;
- a lesson that introduces nothing and is not a project, refresher or review lesson;
- a title, idea or goal that names an item introduced outside its `requires` chain (homonyms in `warns` and items in `mentions` are skipped; a longer known phrase such as "gradient descent" hides the shorter ones inside it);
- a text that names PyTorch or NumPy code while `ladder` lacks that rung;
- a text that names a dataset, pretrained model or corpus while the lesson has no `data` field;
- a core step of the main route that needs a GPU but has no `compute_minutes`;
- a running project with only one step in a route, or a project step that stands alone (it builds on no earlier step of its project and no later step builds on it).

The reports also list, for information: subtopics covered by more than one lesson, project steps that do not end up in the project's final step, and items that no later lesson names or uses yet.

## Reports

- `rapport/controle.md`: errors, warnings, numbers per level, compute time and fallback per GPU step, running projects, code ladder, goal routes.
- `rapport/takken-plan.md`: per branch its topics, size, attach points, where it can start, and the build phases after the main route (what each phase builds ahead from other branches).
- `rapport/kerncijfers.md`: the numbers `Leerroute.md` cites. The checker fills them into `Leerroute.md` between `<!--n:key-->` and `<!--/n-->`, so the design doc never goes stale.

## When the map changes

Subtopics have no ids yet, so a `covers` reference is an index. When the map inserts a subtopic, later indexes shift. The map thread keeps old versions in `onderzoek/kaart/versies/`, and the route keeps its own copy of every version it pins in `data/kaart-pins/`. After a map update:

```sh
python3 tools/migrate_refs.py                    # shows what moves
python3 tools/migrate_refs.py --write            # rewrites covers and the map: key
python3 tools/check_route.py
```

`migrate_refs.py` finds each referenced subtopic in the new map by its exact name. It refuses to write when a subtopic has disappeared, when a reference is invalid, or when a subtopic was only found by a similar name; check those matches and run it again with `--accept-renames`. After writing it reads the files back and restores them if the result differs from its plan, and it pins the new map version in `data/kaart-pins/`. Once the map gives subtopics stable ids, references switch to those ids and this step goes away.
