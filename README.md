# Everything AI

*[Nederlands](README.nl.md)*

Learn AI by building it yourself: from your first line of Python to your own language model and a research paper you reproduce. Every lesson runs on your own computer, and every exercise is checked straight away. Free, in English and Dutch.

## Start

1. Install [uv](https://docs.astral.sh/uv/) once. On macOS and Linux, and in WSL2 on Windows:

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Get the course and start it:

   ```sh
   git clone https://github.com/QRRV/everything-ai.git
   cd everything-ai
   uv run everything-ai
   ```

The course opens in your browser. The first start takes a little longer, because uv fetches the right Python and the packages the course needs. After that, it starts in seconds and works without internet.

To get new lessons later: `git pull` in the same folder. Your own work is kept.

**Windows:** use the course in [WSL2](https://learn.microsoft.com/windows/wsl/install), and run the steps above in the Ubuntu terminal. **macOS:** a Mac with Apple silicon (M1 or newer).

## What you need

The first part of the course runs on any computer from the last few years. Lessons that train bigger models say up front what they need, and what that means on your computer.

## How it works

- **One route, three levels.** The start page shows the whole route, from the basics through intermediate to expert. You can open any lesson; if you miss something from before, the lesson says where to find it.
- **Every lesson works the same way.** Why it matters, the idea with a worked example, then exercises with a check that tells you straight away what is right, what is not yet, and what to do next. Hints open one at a time, with the solution last.
- **Your work stays yours.** You work in your own copy of every lesson, in the folder `work/`. Your progress is kept there too. "Start over" puts a lesson back the way the course made it.
- **English and Dutch.** Switch on the start page or at the top of a lesson.

This is the beginning: the first lessons are ready, and the start page marks the rest as coming soon.

## Using your own AI assistant

There is no AI tutor built in. If you use an assistant such as Claude Code, point it at this folder: [AGENTS.md](AGENTS.md) tells it how the course works and how to help you without giving the answers away.

## Something not right?

Use "Report it on GitHub" at the bottom of the lesson, and say what happened.

## Licences

- Text and figures: [CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt)
- Code in the lessons: [MIT-0](LICENSES/MIT-0.txt)
- The app: [MIT](LICENSES/MIT.txt)
- Bundled fonts (Inter, Source Serif 4, JetBrains Mono): [SIL OFL 1.1](LICENSES/OFL-1.1.txt)

Every file says its licence in its header or in [REUSE.toml](REUSE.toml).
