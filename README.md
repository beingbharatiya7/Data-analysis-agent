# Data Analysis Agent (with persistent memory)

An AI agent that takes **any tabular dataset (CSV / Excel)** and answers plain-English
questions by **running real pandas analysis with its own tools**, not by guessing.
It also has a **persistent memory** that remembers useful facts (definitions, corrections,
preferences, named filters) across sessions and uses them in later answers.

Submission for the BYTE society selection: *The Agentic Task [Basic]* (with the memory deep-dive).

---

## How it works

```
 You ask a question
        |
        v
  Gemini (the "brain") sees the list of tools + any relevant saved notes
        |
        v
  Gemini picks a tool and its arguments  ->  OUR Python code runs it with pandas
        |                                         |
        +<---------- real result sent back -------+
        |
   (repeat if more tools are needed)
        |
        v
  Final answer written from the real results
```

The language model never does the maths. It only decides **which tool to call**. All numbers
come from pandas. Every tool call is printed as a `[tool] ...` line so you can see the
analysis happening.

## The 6 analysis tools (`tools.py`)

| Tool | What it does | Kind of analysis |
|---|---|---|
| `describe_dataset` | Rows, column types, missing values, duplicates, sample rows | Structure & data quality |
| `summary_stats` | Mean, median, std, min, max, quartiles of a numeric column | Distribution |
| `filter_rows` | Rows matching a condition (`>`, `<`, `==`, `!=`, `contains`...), with count and percentage | Selection |
| `group_and_aggregate` | Group by a column and sum / mean / count / min / max / median another | Comparison between categories |
| `correlation` | Pearson correlation between two numeric columns, with plain-language strength | Relationship |
| `plot_chart` | Bar, line, histogram or scatter chart saved as PNG | Visualisation |

They are genuinely different operations, and none is tied to a dataset: columns are passed in
as arguments. Wrong column names return a helpful error (listing the real columns) so the
agent can correct itself.

## Memory design (`memory.py`)

Stored as a small JSON file. It is **not** a chat log and **not** a data dump.

* **What is kept:** only short facts of four kinds: `definition` (what a metric means),
  `correction`, `preference` (how the user likes answers), `filter` (a reusable named condition).
  Values over 300 characters are rejected.
* **Who decides:** the agent decides when something is worth saving (guided by its system
  prompt), via `remember()` and `forget()` tools.
* **Retrieval:** before each question, only notes whose words overlap the question are loaded
  (preferences are always loaded). They are added to the prompt and the agent applies them.
* **Stale or wrong facts:** saving an existing key again **replaces** the old value; the previous
  value is kept in a `previous` field so a conflict is visible.
* **Scope:** each note belongs to one dataset or to all datasets (`*`).
* **User control:** `agent.memory.pretty()` shows everything with usage counts; the user can ask
  the agent to forget a note.

Example (from a real run): after "Remember this: 'big orders' means total_sales greater than
1000", a later question "How many big orders are there?" produced
`[memory] using 1 saved note(s): big orders` and a `filter_rows(total_sales > 1000)` call
-> 198 of 300 orders (66%). A brand-new agent object still used the saved note.

## What I used

* **Language / libraries:** Python, pandas, matplotlib, openpyxl
* **LLM:** Google Gemini via the `google-genai` SDK (free tier), automatic function calling.
  Model names are tried in order (`gemini-3.5-flash`, `gemini-3.8-flash`,
  `gemini-flash-latest`); override with the `GEMINI_MODEL` environment variable.
* **Environment:** Google Colab
* **Memory:** plain JSON file, keyword-overlap retrieval (no extra libraries)

## Setup

**Google Colab (easiest)**
1. Open `data_agent_colab.ipynb` in Colab.
2. Add a Colab secret named `GEMINI_API_KEY` (free key from Google AI Studio).
3. Run the cells from the top.

**On your own computer**
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
python agent.py sample_data/store_sales.csv
```

**Using your own data:** `agent.load("your_file.csv")` then `agent.ask("your question")`.

**Tests without an API key:** `python test_tools.py` and `python test_memory.py`.

## Project files

```
tools.py        the 6 analysis tools
agent.py        the agent loop (Gemini + tools + memory + retries)
memory.py       persistent memory
test_tools.py   runs every tool on two different datasets
test_memory.py  tests save / update / retrieve / forget / persistence
sample_data/    two example datasets (store sales, student performance)
data_agent_colab.ipynb   step-by-step Colab demo
```

## Known limitations

* The free Gemini plan has per-minute and daily limits. The agent waits and retries
  automatically on 429 / 503 errors, then tries the next model.
* Memory relevance uses simple keyword overlap, not embeddings, so a note phrased with
  completely different words may not be retrieved.
* On Colab the memory file is lost if the runtime is reset (set `MEMORY_PATH` to a Google Drive
  path to keep it).
* The agent decides when to save a memory, so it can occasionally miss or over-save a note.
* Chart type is chosen by the model; asking for a "distribution" may not always give a histogram.

## My contribution and AI assistance

* I used **Claude (Anthropic)** as an AI assistant to help design and write the code, and to
  explain each step to me. I used **Gemini** as the model inside the agent.
* I set up the Colab environment and API key, ran every test and demo myself, and checked the
  results (including the memory demo and the retry behaviour on rate-limit errors).
