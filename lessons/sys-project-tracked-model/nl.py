# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-project-tracked-model. Route: course/route/lessen/systemen.yaml. Checks, de uren van 2011, de runs van de course, de demo en de afbeeldingen: checks.py; de bestanden die de course in je project zet: project/ops/demand.py en project/ops/safetensors_io.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("sys-project-tracked-model", "nl")
    C = L.checks   # de uren van 2011, de runs van de course, de checks van je onderdelen, de demo en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een model inpakken: weights, instellingen en een veilig formaat](/course/open/sys-model-package?lang=nl) en [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_package():
    quick_package = L.choice(
        "quick-package",
        question="Je netwerk staat in een safetensors-bestand. Wat staat er in dat bestand?",
        options={
            "pickle": "Het netwerk als Python-objecten, en Python bouwt de objecten bij het laden opnieuw op, de class inbegrepen",
            "right": "Een JSON-header met de naam, het dtype, de shape en de plek van elke tensor, en dan de bytes van alle getallen",
            "settings": "De weights en elke instelling van het model, zoals het aantal neuronen en de learning rate, samen in één header",
            "code": "De code van de class van het netwerk, met de weights erin geschreven als Python-getallen",
        },
        correct="right",
        why={
            "right": "De header zegt waar de getallen van elke tensor staan, en de getallen volgen als onbewerkte bytes: het bestand lezen neemt slices en voert niets uit. De instellingen om het netwerk op te bouwen gaan in een config.json ernaast.",
            "pickle": "Dat schrijft `torch.save`, met pickle: een formaat dat tijdens het laden willekeurige code kan uitvoeren. Wat bevat een safetensors-bestand in plaats daarvan? Zie [Een model inpakken: weights, instellingen en een veilig formaat](/course/open/sys-model-package?lang=nl).",
            "settings": "De header van een safetensors-bestand bevat het dtype, de shape en de plek van elke tensor: waar gaan de class en de instellingen van het netwerk heen? Zie [Een model inpakken: weights, instellingen en een veilig formaat](/course/open/sys-model-package?lang=nl).",
            "code": "Een safetensors-bestand bevat helemaal geen code, en juist dat maakt het laden veilig: wat staat er na de header? Zie [Een model inpakken: weights, instellingen en een veilig formaat](/course/open/sys-model-package?lang=nl).",
        },
    )
    quick_package
    return (quick_package,)


@app.cell(hide_code=True)
def quick_check_val():
    quick_val = L.choice(
        "quick-val",
        question="Je traint vier versies van een model, elk met andere hyperparameters. Op welke data kies je de beste versie?",
        options={
            "train": "Op de training set: de versies leerden ervan, dus daar zie je het best wat elke versie kan",
            "right": "Op de validation set: data waar geen van de versies van leerde, apart gehouden van de test set",
            "test": "Op de test set, de data die je tot het eind bewaart, zodat de keuze meteen op het laatste getal rust",
            "all": "Op alle data tegelijk, training en validatie samen, zodat de keuze op meer rijen rust",
        },
        correct="right",
        why={
            "right": "De validation set zegt hoe elke versie het doet op data waar de versie niet van leerde. De test set blijft buiten elke keuze, zodat het getal aan het eind nog zegt hoe ver de gekozen versie op nieuwe data ernaast zit.",
            "train": "Leert een versie de training set uit het hoofd, dan scoort de versie daar het best en op nieuwe data het slechtst: van welke data leerde geen enkele versie? Zie [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).",
            "test": "Hielp de test set bij het kiezen, dan heeft de test set een keuze gezien en komt het getal te mooi uit: op welke data kies je dan? Zie [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).",
            "all": "Een deel van alle data is de training data, en daar leerden de versies van: op welke data zie je hoe een versie het doet met rijen die de versie nooit zag? Zie [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).",
        },
    )
    quick_val
    return (quick_val,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een bedrijf met deelfietsen heeft overal in een stad stations met fietsen. In de ochtendspits lopen de stations in de buitenwijken leeg en de stations in het centrum vol; busjes rijden de fietsen dan terug, en het bedrijf moet weten hoeveel fietsen mensen de komende uren gaan huren. Een model kan dat zeggen. In dit project, *ops*, bouw je zo'n model in drie stappen, en je beheert het model zoals een team een model beheert: in deze les train je versies, houd je elke versie bij en pak je de beste in; in de volgende stappen test je het model, en zet je het achter een service die bewaakt of de wereld nog lijkt op de data waar het model van leerde.

        De data zijn echt: elk uur van 2011 en 2012 bij Capital Bikeshare in Washington, D.C., met het weer van dat uur, verzameld door Hadi Fanaee-T en João Gama (2013). In [Je runs bijhouden: experiment tracking](/course/open/sys-experiment-tracking?lang=nl) en [Een model inpakken: weights, instellingen en een veilig formaat](/course/open/sys-model-package?lang=nl) hield je de runs van de course bij en pakte je het netwerk van de course in. Nu zijn de onderdelen van jou, en gaan de onderdelen je project in: de splitsing, de bijgehouden run en het pakket.

        **Een voorbeeld uit het dagelijks leven.** Een bakkerij probeert vier recepten voor een nieuw brood. De bakker schrijft elk recept en de uitkomst in een schrift, beoordeelt de broden op dagen waarop de recepten niet zijn bijgesteld, en schrijft dan het winnende recept op een kaart met alles wat een collega nodig heeft om het brood te bakken: de hoeveelheden, de oven, de tijd, en uit welke bakronde het recept komt.
        """),
        draw.project_steps(["Een model van de vraag, bijgehouden en ingepakt", "Tests en een model card", "Je model als service, bewaakt op drift"], 0,
                           here_label="← vandaag", title="Waar je project staat: ops"),
    ])
    return


@app.cell(hide_code=True)
def predict_split():
    guess_split = L.guess(
        "split",
        question="In deze les is de validation set 73 hele dagen van 2011, elke vijfde dag, met alle 1.733 uur. Stel dat je in plaats daarvan 1.733 willekeurige uren nam, van alle dagen van het jaar, en op de rest trainde. Hoe verandert de validation RMSE met de beste instellingen van de course, denk je?",
        options={
            "same": "Ongeveer gelijk: 1.733 uur in beide gevallen",
            "right": "Lager, met ongeveer 7 fietsen per uur",
            "higher": "Hoger, met ongeveer 7 fietsen per uur",
            "much": "Lager, met ongeveer 40 fietsen per uur",
        },
        reveal="**Ongeveer 7 lager.** Gemiddeld over drie seeds is de RMSE op 1.733 willekeurige uren 45,7 fietsen per uur, en op de 73 hele dagen 53,1. Met willekeurige uren heeft het model geleerd van de andere uren van dezelfde dagen, met hetzelfde weer en dezelfde drukke momenten, dus zijn de validatie-uren makkelijker. Een nieuwe dag is nooit zo: op een nieuwe dag zit het model zo'n 53 fietsen ernaast, zoals de hele dagen zeggen.",
    )
    guess_split
    return (guess_split,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De uren.** De course host de tabel als `data/bike-sharing-hour.csv.gz`. Elke rij is één uur. `cnt` is het aantal fietsen dat in dat uur gehuurd is: wat het model moet zeggen. `hr` is het uur van de dag, van 0 tot en met 23; `weekday`, `workingday` en `holiday` zeggen wat voor dag het is; `weathersit` loopt van 1, helder, tot 4, zware regen; `temp`, `atemp` (hoe warm het voelt), `hum` en `windspeed` zijn door de makers van de tabel geschaald tot tussen 0 en 1; `season` en `mnth` geven de tijd van het jaar. Je project gebruikt 2011: 8.645 uur van 365 dagen, want er ontbreken uren. De uren van 2012 wachten op de laatste stap van het project.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        De afbeelding laat zien waarom het uur zo belangrijk is, en waarom het uur op verschillende dagen anders uitpakt. Op een werkdag fietsen mensen naar het werk en terug: één piek om 8 uur 's ochtends, een hogere om 5 uur 's middags. Op een zaterdag is er één brede heuvel in de middag.

        **De inputs en het netwerk.** De course zet `work/projects/ops/demand.py` in je project, met de onderdelen die niet het onderwerp van deze les zijn. `read_hours()` leest de uren van 2011 en voegt een kolom `day` toe: 0 voor 1 januari, tot 364 voor 31 december. `inputs(hours, scale)` maakt 34 getallen per uur. Het uur wordt 24 one-hot-kolommen, zoals in [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl): twee pieken zijn geen rechte lijn, en een kolom per uur laat het netwerk elk uur een eigen weight geven. De andere 10 kolommen zijn gestandaardiseerd, zoals in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl): min het gemiddelde, gedeeld door de standaardafwijking. `scaling(train)` berekent deze gemiddelden en standaardafwijkingen op de trainingsuren. `DemandNet(hidden)` is een netwerk met één hidden layer van `hidden` neuronen: een linear layer van 34 getallen naar `hidden`, ReLU, en een linear layer naar één getal, de fietsen. `train_epoch` traint het netwerk één epoch in mini-batches, met de mean squared error als loss.

        **Hele dagen voor validatie.** De uren van één dag horen bij elkaar: het weer, het soort dag en de drukke momenten zijn hetzelfde. Zitten sommige uren van een dag in de training set en andere in de validation set, dan kent het model de dag al, en zegt de validation set minder over een nieuwe dag: een lek, zoals in [Te mooi om waar te zijn: data leakage](/course/open/ml-data-leakage?lang=nl). Dus gaat de splitsing per dag: elke vijfde dag, de dagen 4, 9, 14 enzovoort, gaat met alle uren naar de validation set. Dat geeft 73 dagen, verspreid over het hele jaar: elk seizoen, elke dag van de week, ook feestdagen.

        **Een versie is een bijgehouden run.** Elke set hyperparameters, zoals 16 neuronen en een learning rate van 0,01, traint een ander netwerk: een andere versie van het model, hier met de naam `h16-lr0.01`. Je `train_version` traint de versie, en meet na elke epoch de root mean squared error op de validatiedagen, zoals in [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl): de wortel uit het gemiddelde gekwadrateerde verschil, in fietsen per uur. De run is een dict met de naam, de hyperparameters en de metrics, en gaat als één regel JSON in het bestand `runs.jsonl`, zoals in [Je runs bijhouden: experiment tracking](/course/open/sys-experiment-tracking?lang=nl).

        **Kies op de laatste epoch.** In *Je runs bijhouden* was de beste run de run met de laagste validation loss bij welke epoch dan ook. Toch geeft `train_version` het netwerk na de laatste epoch terug: het netwerk van een eerdere epoch is weg, tenzij je het toen bewaarde, zoals in een checkpoint. Dus kies je de versie met de laagste validation RMSE bij de laatste epoch: dat is het netwerk dat je hebt.

        **Pak het in.** Je `pack` schrijft de weights van de gekozen versie als `model.safetensors`, met `save_file` uit `work/projects/ops/safetensors_io.py`: het formaat van [Een model inpakken: weights, instellingen en een veilig formaat](/course/open/sys-model-package?lang=nl), dat de course ook in je project zette. De `version` in het pakket is de eerste 12 tekens van de sha256 van de bytes van dat bestand, zoals in [Welk model is dit? Versies en een model registry](/course/open/sys-model-versions?lang=nl): de naam `h16-lr0.01` zegt hoe een versie getraind is, en deze 12 tekens zeggen welk bestand het is. Ernaast komt `config.json`, met wat de weights niet zeggen: de class en het aantal neuronen, hoe de inputs gemaakt worden, de gemiddelden en standaardafwijkingen van de trainingsuren, wat de output betekent, de run waar de versie uit komt met de validation RMSE, en `version`. Bij het laden van het pakket controleer je eerst of `version` klopt met het bestand.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **vraag** | | hoeveel mensen van iets willen in een bepaalde tijd: hier de fietsen die in een uur gehuurd worden | `cnt` |
        | `~` | tilde | draait een mask om: elke True wordt False, en elke False wordt True | `hours[~val_day]` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een model, bijgehouden en ingepakt:

    1. Splits de data zo dat de validation set op nieuwe data lijkt: hier hele dagen, verspreid over het jaar.
    2. Bereken alleen op de training set hoe de inputs gemaakt worden, zoals de gemiddelden en standaardafwijkingen.
    3. Train elke versie met een vaste seed, meet na elke epoch de metric op de validation set, en voeg de run als één regel toe aan het bestand met runs.
    4. Kies de versie met de beste metric bij de laatste epoch, want dat is het netwerk dat je hebt.
    5. Pak de versie in: de weights in een safetensors-bestand, `version` uit de bytes van dat bestand, en een config.json met al het andere wat iemand nodig heeft om het model te gebruiken.
    6. Laad het pakket terug, controleer `version`, en kijk of het pakket dezelfde voorspellingen geeft als het netwerk dat je trainde.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een versie zegt voor vier uren 30, 40, 220 en 110 fietsen; het waren er 10, 40, 250 en 100. De verschillen zijn 20, 0, −30 en 10, de kwadraten 400, 0, 900 en 100, en het gemiddelde van de kwadraten is 350. De validation RMSE is $\sqrt{350} \approx 18{,}7$ fietsen per uur. Het uur met 30 ernaast telt voor meer dan de helft mee: kwadraten laten grote fouten zwaarder tellen.

    Na twee epochs is de run van `h16-lr0.01` één regel in `runs.jsonl`:

    ```json
    {"name": "h16-lr0.01", "params": {"hidden": 16, "lr": 0.01, "epochs": 2, "batch": 64, "seed": 0}, "metrics": {"val_rmse": [[1, 146.87], [2, 103.32]]}}
    ```

    Het laatste paar is `[2, 103.32]`: na epoch 2 zit het netwerk op de validatiedagen 103,32 fietsen per uur ernaast, dus krijgt een pakket van deze versie `"val_rmse": 103.32` in config.json, en niet 146,87.
    """)
    return


@app.cell(hide_code=True)
def text_hours():
    mo.md(r"""
    ### De uren van 2011

    De cel leest de uren met `read_hours` uit je project, telt de uren met `len` en de dagen met `.nunique()`, laat de eerste drie rijen zien met `.head(3)`, en het drukste uur met `.max()`. Voer de cel uit.
    """)
    return


@app.cell
def example_hours():
    from ops.demand import read_hours

    hours = read_hours()   # elk uur van 2011, met een kolom day: 0 voor 1 januari
    print(len(hours), "uur van", hours["day"].nunique(), "dagen")
    print(hours[["dteday", "day", "hr", "weathersit", "temp", "cnt"]].head(3))
    print("drukste uur:", hours["cnt"].max(), "fietsen")
    return


@app.cell(hide_code=True)
def text_exercise_split():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Hele dagen voor validatie

        `split_days(hours)` krijgt een tabel met uren met een kolom `day`, en geeft twee tabellen terug: de training set en de validation set. De `return`-regel staat er al: `hours[val_day]` houdt de rijen waar het mask `val_day` True is, en `~` draait het mask om, dus houdt `hours[~val_day]` alle andere rijen. Vervang de `...` door het mask: True voor elk uur van de dagen 4, 9, 14 enzovoort, tot en met 364. `%` geeft de rest van een deling: `14 % 5` is 4.

        De check roept `split_days(HOURS)` aan, met `HOURS` de 8.645 uur van 2011. De uitkomst hoort een training set van 6.912 uur en een validation set van 1.733 uur te zijn, van de dagen 4, 9, 14 en verder. Daarna probeert de check drie kleine eigen tabellen: tien dagen met één heel druk uur, de laatste tien dagen van het jaar, en dagen waarvan er een paar ontbreken. Is het goed, dan bewaart de check je functie in je project, als `work/projects/ops/split.py`.
        """),
        L.task("Maak `val_day` True voor elk uur van elke vijfde dag, de dagen 4, 9, 14 enzovoort."),
    ])
    return


@app.cell(hide_code=True)
def exercise_split():
    box_split = L.code_box("split", lines=5)
    box_split
    return (box_split,)


@app.cell(hide_code=True)
def check_split(box_split):
    result_split = L.check("split", box_split.value)
    result_split
    return (result_split,)


@app.cell(hide_code=True)
def hints_split():
    L.hints(
        "split",
        [
            "Het mask vergelijkt een kolom van `hours` in één keer voor elke rij met een getal, zoals `hours[\"hr\"] == 8` zou doen. Hier is de kolom de dag, geeft `%` de rest na delen door 5, en is `val_day` True waar de rest 4 is.",
            "Dag 4 is 5 januari, en dag 9 is 10 januari: `4 % 5`, `9 % 5` en `364 % 5` zijn allemaal 4, en `0 % 5`, `5 % 5` en `10 % 5` zijn 0. Alle uren van één dag hebben dezelfde `day`, dus houdt een mask op de dag de uren van een dag bij elkaar.",
            "`    val_day = ___ == 4`",
        ],
        why="`val_day = hours[\"day\"] % 5 == 4`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train(result_split):
    result_split  # opnieuw gemaakt na de check van oefening 1: de check bewaart je splitsing
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een versie als bijgehouden run

        `train_version(name, params, data, log_path, show=None)` traint één versie en houdt de versie bij als run. De eerste regels staan er al: `from ops.demand import DemandNet, train_epoch` haalt het netwerk en één epoch training uit je project, en `range` telt de epochs vanaf 1. `data` bevat vier tensors: `x` en `y`, de inputs en de fietsen van de trainingsuren, en `x_val` en `y_val`, de inputs en de fietsen van de validatie-uren. `torch.manual_seed` maakt de run elke keer gelijk bij dezelfde `params["seed"]`. `DemandNet` maakt het netwerk, `model`, met `params["hidden"]` neuronen, en `torch.optim.Adam` maakt `opt`, een optimizer voor `model.parameters()` met slimmere stappen dan gewone gradient descent, met learning rate `params["lr"]`; een latere les legt Adam uit. `run` is de dict van de run, zoals in [Je runs bijhouden: experiment tracking](/course/open/sys-experiment-tracking?lang=nl). In elke `epoch` traint `train_epoch` het `model` op alle trainingsuren, in mini-batches van `params["batch"]`. Daarna meet `torch.no_grad()` zonder een graph op te bouwen, maakt `float` van `rmse` een getal voor de run met `.append`, en krijgt `show`, als `show` gegeven is, de epoch, het model en de run: de demo tekent daarmee.

        Vervang de twee `...`. Eerst `rmse`: de validation RMSE van `model` op `x_val` en `y_val`, als tensor met één getal. `model(x_val)` en `y_val` hebben allebei de shape (n, 1). `.mean()` geeft het gemiddelde van een tensor, en `.sqrt()` de wortel. Daarna, na de loop, één regel die `run` als één regel JSON aan het open bestand `f` toevoegt, met `json.dumps` en `f.write`. `open(log_path, "a")` opent het bestand op `log_path` om aan het eind toe te voegen.

        De check roept `train_version("check", {"hidden": 16, "lr": 0.01, "epochs": 3, "batch": 64, "seed": 0}, DATA, LOG)` aan, met `DATA` de vier tensors van de splitsing van 2011, en `LOG` een leeg bestand. De run hoort `val_rmse` `[[1, 146.87], [2, 103.32], [3, 87.53]]` te hebben, op twee decimalen, en het bestand één regel. `name` is de naam van de run, hier `"check"`. Daarna probeert de check twee kleine eigen gevallen, één met een bestand waarin al een run staat. Is het goed, dan bewaart de check je functie als `work/projects/ops/train.py`.
        """),
        L.task("Maak `rmse` de validation RMSE van het model, en voeg de run als één regel JSON toe aan het bestand."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train", lines=24)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train, result_split):
    result_split  # opnieuw checken als oefening 1 je splitsing bewaart
    result_train = L.check("train", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train",
        [
            "`rmse` bouw je in drie stappen: de verschillen tussen `model(x_val)` en `y_val`, gekwadrateerd met `**`; dan `.mean()`; dan `.sqrt()`. De regel voor het bestand is `f.write`, met de tekst van `json.dumps(run)` en een regeleinde `\"\\n\"` erachter, zodat de volgende run op een eigen regel begint.",
            "In het uitgewerkte voorbeeld zijn de verschillen 20, 0, −30 en 10, de kwadraten 400, 0, 900 en 100, het gemiddelde 350, en de wortel 18,7. Zonder `.sqrt()` zou de eerste epoch van de check zo'n 21.570 geven in plaats van 146,87: de mean squared error, in fietsen in het kwadraat.",
            "`            rmse = (___).mean().sqrt()`",
        ],
        why="`rmse = ((model(x_val) - y_val) ** 2).mean().sqrt()`, en na de loop `f.write(json.dumps(run) + \"\\n\")`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pack(result_train):
    result_train  # opnieuw gemaakt na de check van oefening 2: de check bewaart je functie
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Pak de gekozen versie in

        `pack(model, run, scale, folder)` zet één versie in een pakket in `folder`. De eerste regels staan er al: `from ops.demand import FEATURES` en `from ops.safetensors_io import save_file` halen twee onderdelen uit je project, `folder.mkdir(parents=True, exist_ok=True)` maakt de map, `weights = folder / "model.safetensors"` is het pad van een bestand in deze map (de `/` zet een map en een naam aan elkaar), `save_file` schrijft `model.state_dict()` naar dit bestand, en `open(weights, "rb")` opent het bestand om de bytes te lezen, als `f`. Daaronder bevat `config` wat iemand nodig heeft om de weights te gebruiken: de class, het aantal neuronen `run["params"]["hidden"]`, hoe de inputs gemaakt worden, de `FEATURES` uit je project, de gemiddelden en standaardafwijkingen in `scale`, wat de output betekent, en de naam van de run. `json.dump` schrijft `config` als `config.json`, met `indent=2`, zodat ook een mens het bestand kan lezen. `hashlib` is de module met de hashes.

        Vervang de twee `...`. `version`: de eerste 12 tekens van de sha256 van de bytes van het bestand, met `hashlib.sha256`, `.hexdigest()` en een slice, zoals in [Welk model is dit? Versies en een model registry](/course/open/sys-model-versions?lang=nl). `val_rmse`: de validation RMSE van de laatste epoch van `run`. `run["metrics"]["val_rmse"]` is de lijst met paren `[epoch, value]`.

        De check roept `pack(MODEL, RUN, SCALE, FOLDER)` aan: `MODEL` is een `DemandNet(16)` zoals het netwerk begint, zodat de check geen training nodig heeft; `RUN` de run van de course van `h16-lr0.01`, met 30 epochs; `SCALE` de gemiddelden en standaardafwijkingen van de trainingsuren van 2011; en `FOLDER` een lege map. De config hoort `"val_rmse": 53.04` te hebben, en als `version` de eerste 12 tekens van de sha256 van `model.safetensors`. Daarna probeert de check twee eigen gevallen: een run van drie epochs, en een run waarvan de laagste RMSE niet bij de laatste epoch ligt. Is het goed, dan bewaart de check je functie als `work/projects/ops/pack.py`.
        """),
        L.task("Maak `version` het begin van de sha256 van het bestand met weights, en `val_rmse` de RMSE van de laatste epoch."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pack():
    box_pack = L.code_box("pack", lines=27)
    box_pack
    return (box_pack,)


@app.cell(hide_code=True)
def check_pack(box_pack, result_train):
    result_train  # opnieuw checken als oefening 2 je functie bewaart
    result_pack = L.check("pack", box_pack.value)
    result_pack
    return (result_pack,)


@app.cell(hide_code=True)
def hints_pack():
    L.hints(
        "pack",
        [
            "Lezen uit het open bestand `f` geeft alle bytes. `hashlib.sha256` maakt daar de hash van, `.hexdigest()` schrijft de hash als 64 tekens, en `[:12]` houdt de eerste 12; de uitkomst gaat in `version`. Voor `val_rmse` geeft `[-1]` het laatste paar van een lijst, en `[1]` de waarde in een paar.",
            "De versie moet veranderen zodra één byte van `model.safetensors` verandert. Het pad van het bestand blijft gelijk voor elk model dat je daar inpakt, en Python print van elke weight alleen een paar cijfers: geen van beide zegt welke bytes je doorgeeft. `RUN` eindigt met het paar `[30, 53.04]`.",
            "`        version = hashlib.sha256(___).hexdigest()[:12]`",
        ],
        why="`version = hashlib.sha256(f.read()).hexdigest()[:12]`, en in de config `\"val_rmse\": run[\"metrics\"][\"val_rmse\"][-1][1],`.",
    )
    return


@app.cell(hide_code=True)
def text_demo(result_pack, result_split, result_train):
    result_split, result_train, result_pack  # opnieuw gemaakt na de checks, die je onderdelen bewaren
    mo.vstack([
        mo.md(r"""
        ### Train vier versies, en pak de beste in

        De knop traint vier versies met jouw onderdelen: je `split_days` maakt de splitsing, `scaling` berekent de gemiddelden en standaardafwijkingen op je trainingsuren, en je `train_version` traint `h16-lr0.001`, `h16-lr0.01`, `h64-lr0.001` en `h64-lr0.01`, elk 30 epochs met seed 0, en voegt elke run toe aan `work/projects/ops/runs.jsonl`. De grafiek toont de validation RMSE van de versie die nu traint, met de versies die al klaar zijn als dunne lijnen van streepjes. Ernaast zie je wat de versie verwacht voor elk uur van donderdag 14 juli 2011, een validatiedag, naast de fietsen die echt gehuurd zijn. Daarna gaat de versie met de laagste RMSE bij de laatste epoch naar je `pack`, en je `pack` schrijft de versie naar `work/projects/ops/model/`; de course leest het pakket terug met een eigen lezer. Als de run klaar is, kun je terug door de epochs.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Train vier versies en pak de beste in")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(result_pack, result_split, result_train, run_button):
    result_split, result_train, result_pack   # opnieuw trainen met je bestanden zodra een check je bestanden bewaart
    trained = None
    if not run_button.value:
        _out = mo.md("*Druk op de knop om de vier versies te trainen.*")
    else:
        _now = {}

        def _show(_name, _epoch, _rmse, _preds, _done):
            if _now.get("name") != _name:   # een nieuwe versie: een eigen grafiek, met de versies die al klaar zijn
                _chart = L.live(C.live_labels(L.lang), C.EPOCHS, role="loss", low=C.LOW, high=C.HIGH, window=1)
                for _other, _points in _done:
                    _chart.reference(_points, _other, role="loss")
                _chart.watch(lambda: C.now_picture(_now, L.lang))
                _now["chart"], _now["name"] = _chart, _name
            _now["epoch"], _now["preds"] = _epoch, _preds
            _now["chart"].add(_epoch, _rmse, C.progress_line(_name, _epoch, _rmse, L.lang))

        mo.output.replace(mo.md("**De uren splitsen** …"))
        _r = C.demo(show=_show)
        if isinstance(_r, str):
            _note = C.broken(_r, L.lang)
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            trained = _r
            _out = mo.vstack([
                L.explore("rewind", "epochs", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.EPOCHS,
                          low=1, high=C.EPOCHS, step=1, role="",
                          task="Ga terug naar epoch 1, en dan vooruit naar 10 en naar 30. Wanneer laat de lijn van wat de beste versie voor 14 juli verwacht voor het eerst twee pieken zien, en welke twee versies zitten bij 30 nog ver boven de andere twee?"),
                mo.Html(C.package_picture(_r, L.lang)),
                mo.md(C.summary(_r, L.lang)),
            ])
    _out
    return (trained,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.vstack([
        mo.md(r"""
        Met de onderdelen van de course eindigen de vier versies op 85,6 (`h16-lr0.001`), 53,0 (`h16-lr0.01`), 71,0 (`h64-lr0.001`) en 53,9 (`h64-lr0.01`) fietsen per uur. `h16-lr0.01` wint, en het pakket bevat 577 getallen in een bestand van 2.596 bytes. Is 53 goed? Zegt een model altijd het gemiddelde van de trainingsuren, dan zit het model er 132,8 fietsen naast; een regel met het gemiddelde van hetzelfde uur op hetzelfde soort dag, werkdag of niet, zit er 79,1 naast: het netwerk is een derde beter dan deze regel. De twee versies met learning rate 0,001 zijn bij epoch 30 nog lang niet klaar: of deze versies met meer epochs zouden bijkomen, zegt deze run niet.

        `h64-lr0.01` was een keer lager: 51,5 na epoch 29. Toch is het netwerk van epoch 29 weg, en het netwerk in je handen zit er 53,9 naast. En de twee versies met learning rate 0,01 liggen binnen één fiets per uur van elkaar, ongeveer zoveel als de seed één versie verschuift: met seed 0, 1 en 2 eindigt `h16-lr0.01` op 53,0, 52,8 en 53,5, en `h64-lr0.01` op 53,9, 53,5 en 53,5. Met seed 0 en 1 is `h16-lr0.01` lager, en met seed 2 eindigen de twee gelijk.
        """),
        L.watch_out("De validatiedagen kozen de versie, dus is de 53,0 een beetje te mooi voor nieuwe dagen: van vier versies nam je de versie die het op deze 73 dagen het best deed. Alleen dagen waarop geen enkele keuze rust laten zien hoe ver het model op een nieuwe dag ernaast zit, zoals de dagen van 2012: deze dagen wachten op de laatste stap van je project."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_scale():
    mo.md(r"""
    ### Oefening 4. De inputs in de service

    In de laatste stap van je project krijgt een service één uur per keer, en maakt de service de 34 inputs met `inputs(hours, scale)` uit je project; pas dan krijgt het netwerk de inputs.
    """)
    return


@app.cell(hide_code=True)
def choice_scale_cell():
    choice_scale = L.choice(
        "scale",
        question="Welke gemiddelden en standaardafwijkingen moet de service aan `inputs` geven als `scale`?",
        options={
            "own": "De gemiddelden en standaardafwijkingen van het uur in het verzoek, zodat de inputs bij dat uur passen",
            "both": "De gemiddelden en standaardafwijkingen van alle uren van 2011 en 2012 samen: meer uren geven betere getallen",
            "right": "De getallen in config.json: de gemiddelden en standaardafwijkingen van de trainingsuren van 2011",
            "none": "Geen: de service kan het netwerk de ruwe waarden geven, want het netwerk leerde wat elke kolom betekent",
        },
        correct="right",
        why={
            "right": "Het netwerk leerde de weights op inputs met precies deze getallen: een warme dinsdag moet in de service dezelfde 34 getallen worden als tijdens de training. Dus schrijft je `pack` de getallen in config.json, naast de weights, en leest de service de getallen daar.",
            "own": "Eén uur heeft één waarde in elke kolom, dus is de standaardafwijking 0, en delen door 0 geeft geen getal; en min het eigen gemiddelde zou elke kolom voor elk uur 0 worden. Met welke getallen werden de inputs gemaakt toen het netwerk leerde?",
            "both": "Met andere getallen wordt dezelfde temperatuur een andere input, en antwoordt het netwerk voor weer dat er niet was. Met welke getallen werden de inputs gemaakt toen het netwerk leerde?",
            "none": "Het netwerk leerde op gestandaardiseerde inputs, de meeste tussen −2 en 2. Een ruwe waarde zoals `mnth` 12 ligt daar ver buiten, waar het netwerk nooit iets leerde. Welke getallen maken de inputs zoals bij de training?",
        },
    )
    choice_scale
    return (choice_scale,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Je runs bijhouden: experiment tracking](/course/open/sys-experiment-tracking?lang=nl), [Welk model is dit? Versies en een model registry](/course/open/sys-model-versions?lang=nl) en [Een model inpakken: weights, instellingen en een veilig formaat](/course/open/sys-model-package?lang=nl) leerde je elk onderdeel op de modellen van de course. Nu heeft je project `ops` alles bij elkaar: een validation set van hele dagen, elke versie als run in `runs.jsonl`, en de beste versie ingepakt met de hash van de weights en de instellingen, klaar om door te geven. De volgende les, *Tests voor ML-code*, geeft je project tests die een bug in trainingscode vangen voordat een lange run begint. In *Project: tests en een model card* test je je model, en in *Project: je model als service, bewaakt op drift* beantwoordt het model verzoeken, en ontmoet het de uren van 2012.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_scale, result_pack, result_split, result_train):
    L.wrap_up(result_split, result_train, result_pack, choice_scale)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
