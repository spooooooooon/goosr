# README for goosr.py

## Description

`goosr.py` is a Python script that you will either love or hate. IRC pal 'spoon' had orginally created this script, and with the depreciation of the completions end point and uncertainty around OpenAI and censorship this script was not maintained.

As recently ChatGPT seems more relaxed and not as censored it now is a good fit for these bots with recent tests. This is my first time coding python, I am sure things could be improved. Eventually we can use locally run models.

The main changes compared to the older version are:

* Will now use the OpenAI Conversations end point.
* Bumped requirements to latest versions.
* `accents` has been changed to `personalities`
  * Each personality is a folder inside `personalities`
  * It must at least contain `system_prompt.txt` as this is the base prompt with each request.
  * For the `goosr` personality we can also load in and vary chat lines to improve the context, this is optional however.
    * `example_chats.txt` will be loaded in if it's found.
    * It will replace hard coded text `### EXAMPLE CHATS` with 300 randomly selected chat lines.
* A default personality can be specified in `goosr.yaml` with `personality`
  * This can be overridden with the `--personality` command line argument.

## Requirements

- Python 3.10 or less
- pip
- venv (Python virtual environment)

## Setup

1. Clone the repository to your local machine.
2. Navigate to the directory containing `goosr.py`.
3. Copy `.env.example` to `.env` and place your OpenAI API or compatible key.

## Virtual Environment Setup

This project uses a Python virtual environment for dependency management. If the virtual environment already exists in the 'venv' folder, you can simply activate it and run your script.

On Unix or MacOS, run:

```bash
source venv/bin/activate
```

On Windows, run:

```bash
.\venv\Scripts\activate
```

Once the virtual environment is activated, you'll see `(venv)` in your shell prompt. This indicates that you're inside the virtual environment.

## Running the Script

You can run `goosr.py` with this command:

```bash
python goosr.py
```

## Command-Line Arguments

`goosr.py` accepts a 'personality' command-line argument. If provided, it overwrites the `globalCfg.personality` value with the provided argument. You can run your script with a 'personality' argument like this:

```bash
python goosr.py --personality new_personality
```

This will overwrite the `globalCfg.personality` value with 'new_personality'. The files and directory must exist.
