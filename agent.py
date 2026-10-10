"""
agent.py - the "brain loop" of the data analysis agent.

How it works:
  1. The user asks a question in plain English.
  2. Gemini sees our 6 tools (from tools.py) and decides which to call.
  3. OUR code runs the tool with pandas and sends the real result back.
  4. Gemini may call more tools, then writes the final answer from the results.

The google-genai SDK runs steps 2-4 automatically ("automatic function
calling"). We add a logging layer so every tool call is recorded and shown,
which proves the answer comes from real analysis and not from model guesses.
"""

import functools
import json
import os
import time

from google import genai
from google.genai import types

import tools
from memory import Memory

# Models are tried in this order. Model names change often, so you can
# override with the GEMINI_MODEL environment variable.
DEFAULT_MODELS = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest"]

SYSTEM_PROMPT = """You are a careful data analyst working on a tabular dataset the user loaded.

Rules:
- ALWAYS use your tools to get numbers. Never guess, estimate, or invent values.
- If you do not know the column names, call describe_dataset first.
- Column names must match exactly. If a tool returns an error, read it and retry with corrected arguments.
- Use several tools when a question needs it (for example: group first, then plot).
- In your final answer: state the result clearly, mention which tools you used,
  and if a chart was saved, give its file path.
- If the question cannot be answered from the data, say so plainly.

Memory rules:
- If the question starts with saved notes, APPLY them (e.g. use a saved
  definition when filtering) and briefly say that you used them.
- Call remember() ONLY when the user states something worth keeping for later:
  what a metric or column means, a correction to a fact, a reusable named
  filter, or a preference about how to answer. Never save raw data, whole
  tables, or ordinary chat. Keep each note to one short sentence.
- If the user says a saved fact is wrong or outdated, call remember() again
  with the same key and the new value. If they ask you to delete something,
  call forget().
"""


class DataAgent:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("No API key. Set GEMINI_API_KEY first.")
        self.client = genai.Client(api_key=key)
        env_model = model or os.environ.get("GEMINI_MODEL")
        self.models = [env_model] if env_model else list(DEFAULT_MODELS)
        self.memory = Memory()
        self.trace: list[dict] = []  # tool calls made during the last question
        self._chat = None
        self._model_in_use = None

    # -- data -----------------------------------------------------------
    def load(self, path: str) -> str:
        """Load a CSV/Excel file and start a fresh conversation."""
        self._chat = None
        return tools.load_dataset(path)

    # -- memory tools (the agent decides when to use them) --------------
    def _memory_tools(self):
        memory = self.memory

        def remember(key: str, value: str, kind: str = "definition",
                     only_this_dataset: bool = True) -> dict:
            """Save ONE short useful fact for future sessions. kind must be:
            'definition' (what a metric/column/term means), 'correction' (a fact
            the user fixed), 'filter' (a reusable named condition) or 'preference'
            (how the user wants answers). key is a short name, e.g. 'active
            customers'. Saving an existing key again REPLACES the old value."""
            dataset = tools.STATE.name if only_this_dataset and kind != "preference" else "*"
            return memory.remember(key, value, kind, dataset)

        def forget(key: str) -> dict:
            """Delete a saved memory by its key, when the user asks to forget it."""
            return memory.forget(key, tools.STATE.name)

        return [remember, forget]

    # -- logging layer around each tool ---------------------------------
    def _logged_tools(self):
        def make(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                self.trace.append({"tool": func.__name__, "args": kwargs, "result": result})
                print(f"  [tool] {func.__name__}({json.dumps(kwargs, default=str)})")
                return result

            return wrapper

        return [make(f) for f in tools.ALL_TOOLS + self._memory_tools()]

    def _new_chat(self, model: str):
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=self._logged_tools(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=8
            ),
        )
        self._model_in_use = model
        return self.client.chats.create(model=model, config=config)

    # -- main entry point -----------------------------------------------
    def ask(self, question: str) -> str:
        if tools.STATE.df is None:
            return "Please load a dataset first."
        self.trace = []
        last_error = None

        # Memory retrieval: pull only the notes relevant to this question.
        picked = self.memory.relevant(
            question, tools.STATE.name, list(tools.STATE.df.columns)
        )
        if picked:
            print(f"  [memory] using {len(picked)} saved note(s): "
                  + ", ".join(it["key"] for it in picked))
            question = self.memory.as_prompt(picked) + "\n\nUser question: " + question

        for model in self.models:
            for attempt in range(3):  # up to 3 tries per model
                try:
                    if self._chat is None or self._model_in_use != model:
                        self._chat = self._new_chat(model)
                    self.trace = []
                    response = self._chat.send_message(question)
                    return response.text or "(The model returned no text.)"
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    # Free-plan limit (429) or Google overloaded (503) -> wait, retry.
                    if any(c in msg for c in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                        last_error = e
                        if attempt < 2:
                            print("  (Google is busy or limit reached, waiting 30 seconds...)")
                            time.sleep(30)
                            continue
                        break  # still limited -> try the next model
                    # Wrong/retired model name -> try the next model.
                    if "404" in msg or "NOT_FOUND" in msg or "not found" in msg.lower():
                        last_error = e
                        self._chat = None
                        break
                    raise
        raise RuntimeError(
            f"None of the models worked: {self.models}. Last error: {last_error}. "
            "Set GEMINI_MODEL to a current model name from Google AI Studio."
        )


if __name__ == "__main__":
    # Tiny command-line demo:  python agent.py sample_data/store_sales.csv
    import sys

    agent = DataAgent()
    print(agent.load(sys.argv[1] if len(sys.argv) > 1 else "sample_data/store_sales.csv"))
    while True:
        q = input("\nAsk a question (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        print("\nAnswer:", agent.ask(q))
