"""
memory.py - persistent memory for the data agent.

Goal (from the task): remember USEFUL facts across sessions, retrieve them when
relevant, and let them change future answers. Not a chat log, not a data dump.

What is stored (each item is small and has a purpose):
  kind = "definition"  e.g. "active customers" -> status != 'cancelled'
  kind = "correction"  e.g. a fact the user fixed
  kind = "preference"  e.g. "give weekly summaries, not daily"
  kind = "filter"      e.g. a reusable named filter "big orders" -> total_sales > 1000

Design choices (explained in the README):
  * JSON file on disk -> survives restarts. Easy to open and read.
  * Items are keyed by a short name, so saving the same key again UPDATES it
    (this is how stale or wrong facts get replaced). The old value is kept in
    'previous' so conflicts are visible, not silent.
  * Each item is scoped to one dataset, or "*" (all datasets). Preferences
    default to "*", definitions default to the current dataset.
  * Only short facts are accepted (max 300 chars), so whole tables or chat
    transcripts cannot be dumped in.
  * Retrieval picks only items relevant to the current question, so memory
    does not clutter every prompt.
  * The user can list and delete anything (inspect / control).
"""

import json
import os
import re
from datetime import datetime

KINDS = {"definition", "correction", "preference", "filter"}
MAX_VALUE_CHARS = 300


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _words(text: str) -> set[str]:
    """Lowercase words, very simple plural stripping, ignore tiny words."""
    out = set()
    for w in re.findall(r"[a-zA-Z0-9_]+", text.lower()):
        if len(w) > 2:
            out.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)
    return out


class Memory:
    def __init__(self, path: str | None = None):
        self.path = path or os.environ.get("MEMORY_PATH", "agent_memory.json")
        self.items: dict[str, dict] = {}
        self._load()

    # -- disk -----------------------------------------------------------
    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.items = json.load(f)

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, indent=2, ensure_ascii=False)

    # -- write ----------------------------------------------------------
    def remember(self, key: str, value: str, kind: str = "definition",
                 dataset: str = "*") -> dict:
        key = key.strip().lower()
        value = value.strip()
        if kind not in KINDS:
            return {"saved": False, "reason": f"kind must be one of {sorted(KINDS)}"}
        if not key or not value:
            return {"saved": False, "reason": "key and value cannot be empty"}
        if len(value) > MAX_VALUE_CHARS:
            return {"saved": False,
                    "reason": f"too long (max {MAX_VALUE_CHARS} characters). "
                              "Memory is for short useful facts, not data."}

        mem_id = f"{dataset}::{key}"
        old = self.items.get(mem_id)
        if old and old["value"] == value:
            return {"saved": True, "status": "already known, no change"}

        item = {
            "key": key, "value": value, "kind": kind, "dataset": dataset,
            "created": old["created"] if old else _now(),
            "updated": _now(),
            "uses": old["uses"] if old else 0,
        }
        if old:  # conflict: keep the previous value so it is visible
            item["previous"] = old["value"]
        self.items[mem_id] = item
        self._save()
        if old:
            return {"saved": True, "status": "updated",
                    "old_value": old["value"], "new_value": value}
        return {"saved": True, "status": "new"}

    def forget(self, key: str, dataset: str = "*") -> dict:
        key = key.strip().lower()
        # try the exact scope first, then any scope with that key
        for mem_id in (f"{dataset}::{key}", *[m for m in self.items if m.endswith(f"::{key}")]):
            if mem_id in self.items:
                del self.items[mem_id]
                self._save()
                return {"forgotten": True, "key": key}
        return {"forgotten": False, "reason": f"nothing stored under '{key}'"}

    def clear(self):
        self.items = {}
        self._save()

    # -- read -----------------------------------------------------------
    def show(self, dataset: str | None = None) -> "list[dict]":
        """Everything stored (optionally only items usable for one dataset)."""
        return [it for it in self.items.values()
                if dataset is None or it["dataset"] in ("*", dataset)]

    def relevant(self, question: str, dataset: str, columns: list[str]) -> list[dict]:
        """Pick only the memories that matter for THIS question."""
        q_words = _words(question)
        picked = []
        for mem_id, it in self.items.items():
            if it["dataset"] not in ("*", dataset):
                continue
            item_words = _words(it["key"] + " " + it["value"])
            overlap = q_words & item_words
            always = it["kind"] == "preference"
            if always or overlap:
                picked.append(it)
                self.items[mem_id]["uses"] += 1
        if picked:
            self._save()
        return picked

    def as_prompt(self, picked: list[dict]) -> str:
        if not picked:
            return ""
        lines = [f"- [{it['kind']}] {it['key']}: {it['value']}" for it in picked]
        return ("Saved notes from earlier sessions. Treat these as the user's own "
                "rules and APPLY them when answering:\n" + "\n".join(lines))

    def pretty(self) -> str:
        if not self.items:
            return "(memory is empty)"
        rows = []
        for it in self.items.values():
            scope = "all datasets" if it["dataset"] == "*" else it["dataset"]
            rows.append(f"* [{it['kind']}] {it['key']} = {it['value']}   "
                        f"(scope: {scope}, used {it['uses']}x, updated {it['updated']})")
        return "\n".join(rows)
