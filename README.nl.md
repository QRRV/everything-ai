# Everything AI

*[English](README.md)*

Leer AI door het zelf te bouwen: van je eerste regel Python tot je eigen language model en een wetenschappelijke paper die je nabouwt. Elke les draait op je eigen computer, en elke oefening wordt meteen nagekeken. Gratis, in het Nederlands en het Engels.

## Starten

1. Installeer één keer [uv](https://docs.astral.sh/uv/). Op macOS en Linux, en in WSL2 op Windows:

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Haal de course op en start hem:

   ```sh
   git clone https://github.com/QRRV/everything-ai.git
   cd everything-ai
   uv run everything-ai
   ```

De course opent in je browser. De eerste keer duurt het wat langer, omdat uv de juiste Python en de pakketten ophaalt die de course nodig heeft. Daarna start hij in een paar seconden en werkt hij zonder internet.

Later nieuwe lessen ophalen: `git pull` in dezelfde map. Je eigen werk blijft bewaard.

**Windows:** gebruik de course in [WSL2](https://learn.microsoft.com/nl-nl/windows/wsl/install), en doe de stappen hierboven in de Ubuntu-terminal. **macOS:** een Mac met Apple silicon (M1 of nieuwer).

## Wat je nodig hebt

Het eerste deel van de course draait op elke computer van de laatste jaren. Lessen die grotere modellen trainen, zeggen vooraf wat ze nodig hebben, en wat dat betekent op jouw computer.

## Hoe het werkt

- **Eén route, drie niveaus.** De startpagina laat de hele route zien, van basis via gevorderd tot expert. Je kunt elke les openen; mis je iets van eerder, dan zegt de les waar je het vindt.
- **Elke les werkt hetzelfde.** Waarom het ertoe doet, het idee met een uitgewerkt voorbeeld, en dan oefeningen met een check die meteen zegt wat goed is, wat nog niet, en wat je nu doet. Hints gaan één voor één open, met de oplossing als laatste.
- **Je werk blijft van jou.** Je werkt in je eigen kopie van elke les, in de map `work/`. Daar wordt ook je voortgang bewaard. "Opnieuw beginnen" zet een les terug zoals de course hem maakte.
- **Nederlands en Engels.** Wissel op de startpagina of bovenaan een les.

Dit is het begin: de eerste lessen zijn klaar, en de startpagina zegt bij de rest dat ze nog komen.

## Je eigen AI-assistent gebruiken

Er zit geen AI-tutor in de course. Gebruik je een assistent zoals Claude Code, wijs hem dan deze map aan: [AGENTS.md](AGENTS.md) vertelt hoe de course werkt en hoe hij je helpt zonder de antwoorden weg te geven.

## Klopt er iets niet?

Gebruik "Meld het op GitHub" onderaan de les, en zeg wat er gebeurde.

## Licenties

- Tekst en figuren: [CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt)
- Code in de lessen: [MIT-0](LICENSES/MIT-0.txt)
- De app: [MIT](LICENSES/MIT.txt)
- Meegeleverde lettertypes (Inter, Source Serif 4, JetBrains Mono): [SIL OFL 1.1](LICENSES/OFL-1.1.txt)

Elk bestand noemt zijn licentie in de kop of in [REUSE.toml](REUSE.toml).
