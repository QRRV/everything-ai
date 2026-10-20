# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-same-on-open-model. Route: course/route/lessen/taal.yaml. Checks and the course's run: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import qwen3
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-same-on-open-model", "nl")
    C = L.checks
    R = C.course()          # the course's run on Qwen3-0.6B-Base, at the settings of a GPU
    Q = qwen3.header()["config"]


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: LoRA, finetunen met een low-rank update](/course/open/llm-lora?lang=nl) en [Project: DPO, voorkeuren zonder reward model](/course/open/llm-dpo?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_frozen():
    quick_frozen = L.choice(
        "quick-frozen",
        question="Na `add_lora(model, 8, 16)` één trainingsstap op een batch: wat verandert er in het model?",
        options={
            "all": "Elke weight, maar elk met een kleinere stap",
            "right": "Alleen de A en B van elke LoRALinear",
            "last": "Alleen de output layer",
        },
        correct="right",
        why={
            "right": "Elke weight die het model had, is bevroren: `add_lora` zet `requires_grad` voor allemaal uit, en alleen de nieuwe A en B krijgen een gradient en een stap.",
            "all": "`add_lora` bevriest elke weight die het model had; de optimizer krijgt alleen de parameters die nog een gradient nodig hebben. Zie [Project: LoRA, finetunen met een low-rank update](/course/open/llm-lora?lang=nl).",
            "last": "De adapters zitten op de lineaire layers van elk blok, niet op de output layer, en de weights van al die layers blijven bevroren. Zie [Project: LoRA, finetunen met een low-rank update](/course/open/llm-lora?lang=nl).",
        },
    )
    quick_frozen
    return (quick_frozen,)


@app.cell(hide_code=True)
def quick_check_reference():
    quick_reference = L.choice(
        "quick-reference",
        question="Je `dpo_step` vergelijkt de policy met een reference model. Wat was dat reference model in het DPO-project?",
        options={
            "copy": "Een kopie van het SFT-model, een tweede keer geladen naast de policy",
            "reward": "Een reward model dat elk antwoord een score geeft",
            "right": "De policy zelf, met zijn adapters uitgeschakeld",
        },
        correct="right",
        why={
            "right": "Met elke adapter uit is het model het model van voor DPO: één model in het geheugen dient als allebei.",
            "copy": "Dat zou ook werken, maar kost een tweede model in het geheugen; je `reference_logprob` zette in plaats daarvan de adapters uit. Zie [Project: DPO, voorkeuren zonder reward model](/course/open/llm-dpo?lang=nl).",
            "reward": "DPO heeft geen reward model nodig: het reference model is de policy van voor de training. Zie [Project: DPO, voorkeuren zonder reward model](/course/open/llm-dpo?lang=nl).",
        },
    )
    quick_reference
    return (quick_reference,)


@app.cell(hide_code=True)
def text_why():
    mo.md(rf"""
    ## Waarom

    Je model heeft {C.number(C.CAPSTONE['params'] / 1e6, 1, 'nl')} miljoen parameters en las verhalen voor jonge kinderen. Grote teams publiceren veel grotere modellen als open weights, met een licentie die zegt wat je ermee mag. Je LoRA, SFT en DPO horen niet bij je model: ze werken op elk model waarvan je de code hebt. Deze les draait ze op Qwen3-0.6B-Base, dat ongeveer 36 biljoen tokens las.

    **Een voorbeeld uit het dagelijks leven.** Een recept van een kok uit het buitenland: de stappen ken je, maar de maten staan in cups in plaats van grammen, en de kruiden heten anders. Je vertaalt dat één keer, en dan kook je zoals altijd.
    """)
    return


@app.cell(hide_code=True)
def predict_rope():
    guess_rope = L.guess(
        "rope",
        question=f"Qwen3 draait zijn queries en keys met RoPE, zoals je ModernGPT, maar het koppelt dimensie i aan i + 64, waar je `rope` de buren 2i en 2i + 1 koppelt. Met zijn eigen indeling is zijn loss op {R['measured']['loss']} apart gehouden antwoorden uit OASST1 {C.number(R['rope']['half'], 2, 'nl')}. De course laadde dezelfde weights in code met jouw indeling. Wat was de loss toen?",
        options={
            "same": "Ongeveer hetzelfde: de hoeken zijn dezelfde, alleen de paren verschillen",
            "bit": "Iets hoger, ongeveer 2",
            "far": "Veel hoger, boven de 4",
        },
        reveal=f"**Veel hoger: {C.number(R['rope']['pairs'], 2, 'nl')}.** De weights leerden welke dimensies samen draaien; met andere paren draait elke query en key op een manier die het model nooit zag. Geen foutmelding, alleen verkeerde getallen. *Een open model heeft dezelfde onderdelen als het jouwe, maar elk detail moet het zijne zijn.*",
    )
    guess_rope
    return (guess_rope,)


@app.cell(hide_code=True)
def text_idea():
    _ours = C.CAPSTONE
    mo.vstack([
        mo.md(rf"""
        ## Het idee

        **Wat een open model is.** Een paar bestanden op een model hub: `config.json` met zijn maten, `model.safetensors` met elke weight onder een naam, de bestanden van de tokenizer, en een licentie. Qwen3-0.6B-Base valt onder Apache-2.0: je mag het gebruiken, veranderen en delen wat je ermee maakt, met zijn licentie, waar het vandaan komt en wat je veranderde.

        **Dezelfde onderdelen.** De blokken van Qwen3 zijn die van [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl): RMSNorm, attention met RoPE en grouped-query attention, en SwiGLU. Wat verschilt, zijn de maten en drie details.

        | | jouw model | Qwen3-0.6B-Base |
        |---|---|---|
        | parameters | {C.number(_ours['params'], 0, 'nl')} | {C.number(R['params'], 0, 'nl')} |
        | blokken | {_ours['layers']} | {Q['num_hidden_layers']} |
        | breedte $d$ | {_ours['d']} | {Q['hidden_size']} |
        | heads, waarvan key-value | {_ours['heads']}, {_ours['kv_heads']} | {Q['num_attention_heads']}, {Q['num_key_value_heads']} |
        | getallen per head | {_ours['d'] // _ours['heads']}, $d$ gedeeld door de heads | {Q['head_dim']}, een eigen getal |
        | tokens in de vocabulary | {C.number(_ours['vocab'], 0, 'nl')} | {C.number(Q['vocab_size'], 0, 'nl')} |
        | tokens waarop het trainde | {C.number(_ours['tokens'] / 1e6, 1, 'nl')} miljoen | ongeveer 36 biljoen |
        """),
        mo.Html(C.sizes_picture({"title": ["parameters", "tokens waarop het trainde"], "rows": ["jouw model", "Qwen3-0.6B"],
                                 "alt": f"Twee staafdiagrammen op een logaritmische schaal. Parameters: jouw model {C.number(_ours['params'] / 1e6, 1, 'nl')} miljoen, Qwen3-0.6B {C.number(R['params'] / 1e6, 0, 'nl')} miljoen. Tokens waarop het trainde: jouw model {C.number(_ours['tokens'] / 1e6, 0, 'nl')} miljoen, Qwen3-0.6B ongeveer 36 biljoen."})),
        mo.md(rf"""
        Op deze logaritmische schalen: ongeveer 190 keer zoveel parameters, en meer dan een half miljoen keer zoveel tokens. De drie details:

        1. **QK-norm.** Vóór RoPE krijgen de queries en keys van elke head een eigen RMSNorm, die de scores van attention stabiel houdt tijdens het trainen.
        2. **De indeling van RoPE.** Dimensie $i$ van een head draait met dimensie $i + 64$, niet met zijn buur.
        3. **Weight tying.** De output layer is weer de token embedding, zoals in [Weight tying](/course/open/lm-weight-tying?lang=nl).

        **De namen.** Het bestand noemt een weight zoals de code van Qwen dat doet, `model.layers.0.self_attn.q_proj.weight`; de Qwen3 van de course gebruikt de namen van je ModernGPT, `blocks.0.attention.query.weight`, zodat je `add_lora` zijn layers vindt. Laden is elke tensor een nieuwe naam geven, en elke shape nagaan.

        **Tokens die het nooit leerde.** Qwen schrijft een gesprek in ChatML, zoals in [Chat templates](/course/open/llm-chat-template?lang=nl): `<|im_start|>user`, de vraag, `<|im_end|>`, dan `<|im_start|>assistant` en het antwoord. Maar in de embedding van het base model heeft `<|im_end|>` precies de getallen van `<|im_start|>` en {R['untrained']['tokens'] - 2} andere rijen. Met weight tying krijgen ze allemaal dezelfde logit, dus kiest greedy schrijven de eerste van hen, een teksttoken. Het enige special token dat het leerde, is `<|endoftext|>`: de course sluit elk antwoord daarmee af.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent |
        |---|---|
        | **`config.json`** | het bestand van een open model met zijn maten: breedte, blokken, heads, vocabulary |
        | **QK-norm** | een RMSNorm op de queries en op de keys van elke head, vóór RoPE |
        | **ongetraind token** | een token dat het model nooit leerde: zijn rij van de embedding heeft precies de getallen van andere ongetrainde rijen |
        """),
        L.watch_out("Ga na welke special tokens een base model leerde, voor je het finetunet. Een ongetraind token kan nooit waarschijnlijker worden dan zijn dubbelgangers, en adapters in de blokken veranderen dat niet: train ook zijn rij van de embedding, of eindig met een token dat het model kent."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een open model aanpassen:

    1. Download zijn bestanden op een vaste versie; lees zijn `config.json` en zijn licentie.
    2. Geef elke tensor uit zijn bestand een naam uit je code, ga elke shape na, en laad.
    3. Schrijf elke vraag in zijn chat template, en sluit elk antwoord af met een token dat het leerde.
    4. SFT met adapters: `add_lora`, `sft_example`, `sft_batch` en `pretrain_step`.
    5. Voeg ze samen met `merge_lora`; DPO met nieuwe adapters en `dpo_step`, de adapters uit als reference model.
    6. Meet op apart gehouden vragen en paren, voor en na elke stap.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De namen en shapes van blok 0, in het bestand van Qwen3 en in het model van de course. De header van het bestand zit in de course, dus dit werkt zonder de download. Voer de cel uit.
    """)
    return


@app.cell
def example_names():
    _file = C.header()["tensors"]
    _ours = qwen3.model_names()
    print("in het bestand van Qwen3:")
    for _name in _file:
        if _name.split(".")[:3] == ["model", "layers", "0"]:
            print("   ", _name, _file[_name]["shape"])
    print("in het model van de course:")
    for _name in _ours:
        if _name.split(".")[:2] == ["blocks", "0"]:
            print("   ", _name, _ours[_name])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(rf"""
    Dezelfde 11 tensors per blok, in een andere volgorde en onder andere namen: `self_attn.q_proj` is `attention.query`, `input_layernorm` is `norm1`, `q_norm` houdt zijn naam. Een weight is (outputs, inputs): de query van `[2048, 1024]` geeft {Q['num_attention_heads']} heads van {Q['head_dim']}. Het bestand heeft geen output layer: die is weer de embedding.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rename():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Geef de weights hun namen

        Maak `PARTS` af, de delen van een naam die verschillen, en `rename(name)`: die laat het eerste deel weg, `model`, en geeft elk deel uit `PARTS` zijn nieuwe naam; andere delen, zoals het nummer van een blok of `q_norm`, blijven zoals ze zijn. De check probeert elke soort naam uit het bestand.
        """),
        L.task("Maak `PARTS` en `rename` af."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rename():
    box_rename = L.code_box("rename", lines=22)
    box_rename
    return (box_rename,)


@app.cell(hide_code=True)
def check_rename(box_rename):
    result_rename = L.check("rename", box_rename.value)
    result_rename
    return (result_rename,)


@app.cell(hide_code=True)
def hints_rename():
    L.hints(
        "rename",
        [
            "De zes lineaire layers die nog niet in `PARTS` staan, zijn `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` en `down_proj`; de tweede lijst van het uitgewerkte voorbeeld heeft hun namen in het model van de course.",
            "`name.split(\".\")` geeft de delen; `[1:]` laat het eerste weg; `\".\".join(parts)` maakt er weer één naam van.",
        ],
        why="Het bestand en de code bevatten dezelfde tensors onder de namen die hun makers kozen; een tabel van de delen die verschillen, maakt van het ene het andere.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_params():
    mo.vstack([
        mo.md(rf"""
        ### Oefening 2. Wat LoRA op Qwen3 traint

        Maak `lora_parameters(config, r)` af: hoeveel getallen je `add_lora` bij rang `r` traint op een model met deze `config.json`. Hij zet een LoRALinear op de zeven lineaire layers van elk blok, en elke traint `A`, `r` bij zijn inputs, en `B`, zijn outputs bij `r`. Lees elke maat uit `config`. De check probeert Qwen3-0.6B, -1.7B en -4B, en een klein model.
        """),
        L.task("Maak `lora_parameters` af."),
    ])
    return


@app.cell(hide_code=True)
def exercise_params():
    box_params = L.code_box("lora-parameters", lines=16)
    box_params
    return (box_params,)


@app.cell(hide_code=True)
def check_params(box_params):
    result_params = L.check("lora-parameters", box_params.value)
    result_params
    return (result_params,)


@app.cell(hide_code=True)
def hints_params():
    L.hints(
        "lora-parameters",
        [
            "De queries zijn `num_attention_heads` heads van `head_dim` getallen; de keys en de values zijn er `num_key_value_heads`.",
            "Voor elke layer `r * n_in + n_out * r`; zeven layers per blok, en `num_hidden_layers` blokken.",
        ],
        why="LoRA traint twee smalle matrices per layer, dus wat het kost, groeit met de inputs plus de outputs van een layer, niet met hun product, zoals bij full fine-tuning.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_end():
    mo.md(r"""
    ### Oefening 3. Waar een antwoord eindigt
    """)
    return


@app.cell(hide_code=True)
def choice_end_cell():
    choice_end = L.choice(
        "end-id",
        question="Je `sft_example` zet `end_id` achter elk antwoord, en de chatloop stopt bij dat token. Je `add_lora` laat de embedding, die ook de output layer is, bevroren. Met welk token kan SFT Qwen3-0.6B-Base leren zijn antwoorden af te sluiten?",
        options={
            "im_end": "`<|im_end|>`: SFT leert het net als elk ander token, zoals de template vraagt",
            "right": "`<|endoftext|>`, het special token dat in de pretraining elke tekst afsloot",
            "any": "Allebei, als elk voorbeeld maar met hetzelfde eindigt",
        },
        correct="right",
        why={
            "right": f"Het enige special token dat het base model leerde. `<|im_end|>` deelt zijn rij met {R['untrained']['tokens'] - 1} andere tokens, dus zijn logit is altijd de hunne.",
            "im_end": f"`<|im_end|>` heeft precies de getallen van {R['untrained']['tokens'] - 1} andere tokens in de bevroren output layer, dus het kan nooit waarschijnlijker zijn dan zij. Kijk nog eens naar het blok *Let op* in *Het idee*.",
            "any": f"Niet voor dit model: `<|im_end|>` deelt zijn rij van de bevroren output layer met {R['untrained']['tokens'] - 1} andere tokens. Kijk nog eens naar het blok *Let op* in *Het idee*.",
        },
    )
    choice_end
    return (choice_end,)


@app.cell(hide_code=True)
def text_demo():
    _m, _k = R["measured"], C.RANK
    mo.vstack([
        mo.md(rf"""
        ### Train Qwen3 met je eigen stappen

        De demo laadt Qwen3 met je `rename`, zet er met je `add_lora` adapters van rang {_k} op, en traint die met SFT op OASST1: vragen met het antwoord dat mensen het beste vonden. Hij voegt ze samen met je `merge_lora`, en traint nieuwe adapters met je `dpo_step` op HelpSteer2: twee antwoorden, waarvan mensen zeiden welk beter was. Ervoor en erna meet hij op apart gehouden vragen en paren.

        Op een GPU doet hij de stappen van de course: {C.SFT['steps']['gpu']} stappen SFT op {C.SFT['batch']['gpu']} vragen en {C.DPO['steps']['gpu']} stappen DPO op {C.DPO['batch']['gpu']} paren. Op een CPU doet hij er {C.SFT['steps']['cpu']} en {C.DPO['steps']['cpu']}, op {C.SFT['batch']['cpu']} tegelijk: ongeveer {C.number(R['cpu']['minutes'], 0, 'nl')} minuten en {C.number(R['cpu']['memory_gb'], 0, 'nl')} GB, en zo weinig stappen DPO blijven op toeval ({R['cpu']['dpo']['agree']} van de {R['cpu']['dpo']['pairs']} apart gehouden paren). De run bewaart het model niet.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B-Base (1,2 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Qwen3-0.6B-Base staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, van `Qwen/Qwen3-0.6B-Base` op een vaste versie: 1,2 GB, zijn weights en zijn tokenizer. De oefeningen hebben het niet nodig.*",
        "busy": "**Bezig met downloaden:** {done} van {total} MB …",
        "no_net": "*De download lukte niet: Hugging Face was niet bereikbaar, of de verbinding brak af. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: het bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is op je schijf geen plek voor 1,2 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B-Base is gedownload.*")
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train Qwen3 met SFT en DPO")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(box_rename, result_rename, train_button):
    trained = None
    if result_rename.state != "correct":
        _out = mo.md("*Maak eerst `rename` goed in oefening 1: de demo laadt Qwen3 ermee.*")
    elif not C.downloaded():
        _out = mo.md("*Download eerst Qwen3, met de knop hierboven.*")
    elif not train_button.value:
        _out = mo.md("*Druk op de knop om Qwen3 te trainen.*")
    else:
        _phases = {"base": "Qwen3-0.6B-Base antwoordt", "sft": "Na SFT antwoordt het", "dpo": "Na DPO antwoordt het", "agree": "Paren gemeten"}

        def _show(_d):
            if _d["phase"] == "load":
                mo.output.replace(mo.md("**Qwen3 laden** en meten …"))
            elif "step" in _d:
                mo.output.replace(mo.md(f"**{_d['phase'].upper()}, stap {_d['step']} van {_d['steps']}:** loss {C.number(_d['loss'], 3, 'nl')}, {_d['seconds']:.0f} s"))
            else:
                mo.output.replace(mo.md(f"**{_phases[_d['phase']]}:** {_d['done']} van {_d['total']} …"))

        _r = C.run(run_code(box_rename.value).names["rename"], _show)
        _out = mo.md({
            "names": "*Je `rename` geeft niet elke tensor uit het bestand een naam van het model, met dezelfde shape. Slaagt hij nog voor zijn check?*",
            "run": "*Het laden van Qwen3, of een van je modules in `work/projects/llm/`, stopt met een fout. Slagen je modules nog voor hun checks? Verwijder een bestand dat je veranderde, en check zijn oefening opnieuw.*",
            "lora": "*Je `add_lora` stopt met een fout op Qwen3, of laat niets over om te trainen. Slaagt hij nog voor zijn check in de les over LoRA?*",
            "sft": "*Je `sft_example`, `sft_batch` of `pretrain_step` stopt met een fout op Qwen3, of de loss is geen getal meer. Slagen ze nog voor hun checks?*",
            "merge": "*Je `merge_lora` stopt met een fout op Qwen3. Slaagt hij nog voor zijn check in de les over LoRA?*",
            "dpo": "*Je `dpo_step`, `answer_logprob` of `reference_logprob` stopt met een fout op Qwen3, of geeft geen getal. Slagen ze nog voor hun checks?*",
            "memory": "*Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop.*",
        }[_r]) if isinstance(_r, str) else mo.md("")
        trained = None if isinstance(_r, str) else _r
    _out
    return (trained,)


@app.cell(hide_code=True)
def demo_result(trained):
    _run = trained or R
    _mine = trained is not None
    _b, _s, _d = _run["base"], _run["sft"], _run["dpo"]
    _whose = "Jouw run" if _mine else "De run van de course"
    _stand = "" if _mine else "\n\n*De run van de course staat hier tot je Qwen3 zelf traint.*"

    def _answer(_text, _ended):
        _t = C.excerpt(_text, 400)
        return (_t or "*(niets)*") + ("" if _ended else " *(nog geen einde)*")

    mo.vstack([
        mo.md(f"**{_whose}**, {_run['steps']['sft']} stappen SFT en {_run['steps']['dpo']} DPO op {_run['device']}, {C.number(_run['seconds']['all'] / 60, 0, 'nl')} minuten. LoRA trainde {C.number(_run['trainable'], 0, 'nl')} getallen.\n\n"
              f"| apart gehouden van de training | Qwen3-0.6B-Base | na SFT | na DPO |\n|---|---|---|---|\n"
              f"| loss op de antwoorden van mensen | {C.number(_b['loss'], 3, 'nl')} | {C.number(_s['loss'], 3, 'nl')} | |\n"
              f"| antwoorden die uit zichzelf ophouden | {_b['stopped']} van {_b['total']} | {_s['stopped']} van {_s['total']} | {_d['stopped']} van {_d['total']} |\n"
              f"| paren waarop de reward van DPO het met de mensen eens is | | | {_d['agree']} van {_d['pairs']} |{_stand}"),
        mo.md(f"**De eerste vraag waarop het nooit trainde:** *{C.excerpt(_run['questions'][0], 300)}*\n\n"
              f"- Qwen3-0.6B-Base: {_answer(_b['texts'][0], _b['ended'][0])}\n"
              f"- na SFT: {_answer(_s['texts'][0], _s['ended'][0])}\n"
              f"- na DPO: {_answer(_d['texts'][0], _d['ended'][0])}"),
    ])
    return


@app.cell(hide_code=True)
def text_after_demo():
    _b, _s, _d = R["base"], R["sft"], R["dpo"]
    mo.md(rf"""
    **De run van de course lezen.** SFT leerde Qwen3 de vorm van een gesprek: {_s['stopped']} van de {_s['total']} antwoorden houden nu uit zichzelf op, waar dat er {_b['stopped']} waren. De loss op de antwoorden van mensen daalde maar een beetje, van {C.number(_b['loss'], 2, 'nl')} naar {C.number(_s['loss'], 2, 'nl')}: het wist al hoe zulke antwoorden gaan.

    DPO schoof het op naar wat mensen verkiezen: zijn reward is het op {_d['agree']} van de {_d['pairs']} apart gehouden paren met de mensen eens, waar toeval de helft geeft, en op {_d['seen']['agree']} van de {_d['seen']['pairs']} paren waarop het trainde. Maar nog maar {_d['stopped']} van de {_d['total']} antwoorden houden binnen {C.NEW} tokens op: de gekozen antwoorden zijn gemiddeld iets langer, en van DPO is bekend dat het antwoorden langer maakt. *Meet meer dan waarop je traint.*
    """)
    return


@app.cell(hide_code=True)
def text_exercise_results():
    mo.md(r"""
    ### Oefening 4. Wat elke stap leerde
    """)
    return


@app.cell(hide_code=True)
def explain_results():
    explain_results_cell = L.explain(
        "results",
        question="Wat leerde SFT Qwen3 in de run van de course, en wat leerde DPO het? En waarom waren zo weinig stappen genoeg, waar je eigen model er duizend nodig had?",
        model_answer=f"SFT leerde het de vorm van een gesprek: antwoorden na de generation prompt en stoppen met `<|endoftext|>` ({R['sft']['stopped']} van de {R['sft']['total']} antwoorden, waar dat er {R['base']['stopped']} waren). Wat het moest zeggen, hoefde het niet te leren: uit ongeveer 36 biljoen tokens kende het de taal al, dus waren {R['steps']['sft']} stappen genoeg en daalde de loss maar een beetje; mijn model moest de taal en de taak tegelijk leren. DPO schoof het op naar wat mensen verkiezen: {R['dpo']['agree']} van de {R['dpo']['pairs']} apart gehouden paren tegen de helft bij toeval, en {R['dpo']['seen']['agree']} van de {R['dpo']['seen']['pairs']} paren waarop het trainde, dus het leerde de paren beter dan de voorkeur. Het veranderde ook iets waarop het niet trainde: minder antwoorden hielden op, {R['dpo']['stopped']} in plaats van {R['sft']['stopped']}. Ik zou op meer paren trainen, en ook de lengte meten.",
        points=[
            "SFT leerde de vorm van een gesprek, het antwoord en waar het eindigt, geen nieuwe kennis",
            "een model dat op biljoenen tokens getraind is, heeft weinig stappen nodig, want het kent de taal al",
            "de reward van DPO is het met mensen eens op de meeste apart gehouden paren, en op bijna alle paren waarop het trainde",
            "DPO veranderde ook iets waarop het niet trainde: minder antwoorden hielden op, dus meet meer dan waarop je traint",
        ],
    )
    explain_results_cell
    return (explain_results_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De blokken van [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl), de adapters van [Project: LoRA, finetunen met een low-rank update](/course/open/llm-lora?lang=nl) en de stappen van [Project: DPO, voorkeuren zonder reward model](/course/open/llm-dpo?lang=nl) werkten op een model dat je niet zelf trainde, zodra zijn namen, details en template de zijne waren. Noem, voor je deelt wat je trainde, de bronnen: Qwen3-0.6B-Base en OASST1 onder Apache-2.0, HelpSteer2 onder CC BY 4.0.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_rename, result_params, choice_end, explain_results_cell):
    L.wrap_up(result_rename, result_params, choice_end, explain_results_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
