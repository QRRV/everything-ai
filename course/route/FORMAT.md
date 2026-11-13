# Route data format

This folder describes the course as data: which lessons exist, in which order a learner meets them, and where every topic of the map lives. Nothing here is a lesson yet; it is the plan the lessons will be built from. `tools/check_route.py` checks all of it and writes `rapport/`.

There are four layers:

1. **The map** (`onderzoek/kaart/kaart_v2.yaml`, owned by the map thread) says *what exists*: 27 map domains, 623 topics, 3,848 subtopics, each with a level N1 to N5 and prerequisite topics.
2. **The structure** (`domeinen.yaml`, besluit 1.18) says *how the course is divided*: the foundation (fundament), whose kern modules everyone takes first and whose shared modules a domain pulls in when it needs them, and 18 equal domains, each with modules, a running project and a final project. Every map subtopic belongs to exactly one module.
3. **Lessons** (`lessen/*.yaml`) say *how it is taught*: one idea per lesson, with its place in the structure, what it needs and what it introduces.
4. **Routes** (`routes.yaml`) say *in which order*: the kern, then a route per domain. A lesson can be in many routes.

## Lesson files: `lessen/*.yaml`

```yaml
map: v2.2          # map version that the "covers" references point into
lessons:
- id: py-first-program
  title: "Your first program"
  level: basis
  domein: fundament
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
| `id` | yes | Stable id in English kebab-case, never a number. Never renamed once lessons exist; the app stores progress by it. Prefix shows the subject (`py-`, `np-`, `ml-`, `nn-`, `torch-`, `cnn-`, `attn-`, `lm-`, `tok-`, `gpt-`, `llm-`, `sys-`, `res-`, `repro-`, `math-`, `prob-`, `stats-`, `info-`, `opt-`, `data-`, `rai-`, `rl-`, `search-`, `start-`, `cv-`). |
| `title` | yes | English title. The Dutch title is written with the lesson. |
| `level` | yes | `basis`, `gevorderd` or `expert`. |
| `domein` | yes | The part of the course the lesson belongs to: `fundament` or one of the domain ids in `domeinen.yaml`. |
| `module` | yes | A module of that part in `domeinen.yaml`. The app shows consecutive lessons of one module as a chapter, named in `../display.yaml` (key `domein/module`). |
| `idea` | yes | The one idea of the lesson, in one or two sentences. If it needs "and also", it is two lessons. |
| `covers` | yes | Map subtopics this lesson teaches, as `topic-id#index` into the map version in `map:`. `[]` means no subtopic fits, on purpose. |
| `requires` | yes | Lesson ids that must come before. Only the first lesson has none. |
| `introduces` | no | Inventory items this lesson explains for the first time. Each item is introduced by exactly one lesson. Symbols (`s:`) are introduced where the notation first appears, even when the termen list already defines them. |
| `uses` | no | Items this lesson uses without explaining them. Each must be introduced somewhere in its `requires` chain. Filled in fully when the lesson is built. |
| `mentions` | no | Items the lesson names in passing without relying on them, such as "your first language model" in the letter-count project. The name check skips them. |
| `warns` | no | Words that mean something else in daily life or in another field (`token`, `kernel`, `bias`, `class`); the lesson says so. The name check treats them as homonyms. |
| `ladder` | no | Code rungs used: `python` (by hand), `numpy`, `torch`. |
| `kind` | no | `project` for a step of a running project; `refresher` for a short repair lesson on something taught earlier; `review` for mixed practice of earlier lessons, including a standalone mini-project. Refresher and review lessons introduce nothing new. |
| `project` | with kind project | Running project id from `domeinen.yaml` (`nn`, `cnn`, `llm`, `repro`, ...). A project step can sit in another part than the project's owner. |
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

A route has an `id`, titles, `milestones` (shown on the start page under the level they end) and `steps`. The first route is the kern (`id: kern`): the start everyone takes before choosing a domain. Every later route is the route of one domain (`domein`) and starts after the kern: the core steps of the kern count as done, and every other lesson its steps need is a step of the route itself (rule 9), borrowed from a shared module of the foundation or from another domain. A domain has at most one route; a domain without one yet has its first lessons in the route that uses them. A step is a lesson id, or `{id: ..., extra: true}` for an optional deepening step (verdieping) in that route. Being extra is a property of the step in a route, not of the lesson: the same lesson can be core in one route and extra in another.

The order of the whole course is the kern, then the other routes in the order of the file, each lesson at its first place. `goals` lists example goal routes. They are computed, not written: the lessons a goal lesson needs, in that order.

## The structure: `domeinen.yaml`

The decided structure of the course (besluit 1.18), in Dutch; the comments at the top of the file describe every field. `fundament` has modules; those with `kern: true` form the kern, the others are shared modules. Every domain has a group on the start page (`groepen`), modules, a running `project` (an `id` that project lessons name, a `titel`, what it yields per level, and `eindles` once the lesson of the final project exists), a final project, papers, the kern lessons that are a taster of it (`proef`) and the shared modules it pulls in per level (`gebruikt`). Each module claims map topics, levels of topics or single subtopics (`onderwerpen`). The structure check (`tools/check_structuur.py`, run by `tools/check_route.py`) holds it to the map and the lessons.

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
9. Every step of a route exists, appears once, comes after everything it requires, and the route contains everything its steps require; for the route of a domain, the core steps of the kern count as contained.
10. A core step of a route never requires an extra step of that route.
11. No route goes down a level.
12. Every lesson's `domein` is `fundament` or a domain of `domeinen.yaml`, and its `module` is a module of that part. The first route is the kern; every other route names its domain, and a domain has at most one route. Every lesson is in a route; every lesson of the kern is a step of the kern route, and every core step of the kern route is a lesson of the kern (a lesson the kern borrows is extra); every lesson of a domain with a route is a step of that route.
13. Every `data` and `reference_data` id is in `bronnen.yaml`; papers point to existing lessons and name an existing `domein`; goals are in a route; every part has a project with an `id` and a `titel`, project ids are unique, and an `eindles` is an existing lesson of that part; every own-choice paper has `needs` and a compute estimate, and at least one runs on a CPU.
14. `kind`, `project`, `minutes`, `compute_minutes`, `fallback`, `hard`, `goal` and `look_back` have valid values; `kind: project` has a `project` and the other way round; a core step of any route computes at most 15 minutes; a core step that needs a GPU has a `fallback` (unless its time depends on the chosen paper); `look_back` points into the `requires` chain.
15. A project step that trains on data with `checkpoints: false` has `reference_data` with data the course may train its handed-out checkpoints on, or `[]`.

Warnings:

- a lesson without a `covers` field;
- a lesson that introduces nothing and is not a project, refresher or review lesson;
- a title, idea or goal that names an item introduced outside its `requires` chain (homonyms in `warns` and items in `mentions` are skipped; a longer known phrase such as "gradient descent" hides the shorter ones inside it);
- a text that names PyTorch or NumPy code while `ladder` lacks that rung;
- a text that names a dataset, pretrained model or corpus while the lesson has no `data` field;
- a core step that needs a GPU but has no `compute_minutes`;
- a running project with only one step in a route, or a project step that stands alone (it builds on no earlier step of its project and no later step builds on it).

The reports also list, for information: subtopics covered by more than one lesson, project steps that do not end up in the project's final step, and items that no later lesson names or uses yet.

## Reports

- `rapport/controle.md`: errors, warnings, lessons per part, every route per level, compute time and fallback per GPU step, running projects, code ladder, goal routes.
- From the structure check: `rapport/dekking.md` (where every map subtopic goes), `rapport/omvang.md` (size per part and the estimate of lessons still to build), `rapport/verbanden.md` (what the parts need from each other), `rapport/lessen-per-deel.md`, `rapport/dubbel.md` and `rapport/domeinen-detail.md`. The structure check's own rules are in its docstring.

`Leerroute.md` is the design document of the first route, from before besluit 1.18; its numbers are no longer kept up to date.

## When the map changes

Subtopics have no ids yet, so a `covers` reference is an index. When the map inserts a subtopic, later indexes shift. The map thread keeps old versions in `onderzoek/kaart/versies/`, and the route keeps its own copy of every version it pins in `data/kaart-pins/`. After a map update:

```sh
python3 tools/migrate_refs.py                    # shows what moves
python3 tools/migrate_refs.py --write            # rewrites covers and the map: key
python3 tools/check_route.py
```

`migrate_refs.py` finds each referenced subtopic in the new map by its exact name. It refuses to write when a subtopic has disappeared, when a reference is invalid, or when a subtopic was only found by a similar name; check those matches and run it again with `--accept-renames`. After writing it reads the files back and restores them if the result differs from its plan, and it pins the new map version in `data/kaart-pins/`. Once the map gives subtopics stable ids, references switch to those ids and this step goes away.
