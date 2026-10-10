"""Tests the memory logic alone (no API key needed)."""
import os, tempfile
from memory import Memory

p = os.path.join(tempfile.mkdtemp(), "m.json")
m = Memory(p)

print(m.remember("active customers", "status is not 'cancelled'", "definition", "customers.csv"))
print(m.remember("summary style", "give weekly summaries, not daily", "preference"))
print(m.remember("big orders", "total_sales > 1000", "filter", "store_sales.csv"))

print("\nDuplicate:", m.remember("active customers", "status is not 'cancelled'", "definition", "customers.csv"))
print("Conflict/update:", m.remember("active customers", "status is 'active' or 'trial'", "definition", "customers.csv"))
print("Too long:", m.remember("dump", "x" * 500))
print("Bad kind:", m.remember("a", "b", "chat"))

print("\nRELEVANT for customers.csv, 'how many active customers?'")
for it in m.relevant("how many active customers do we have?", "customers.csv", []): print("  ", it["key"], "->", it["value"])
print("RELEVANT for store_sales.csv, 'average discount?' (big-orders is not mentioned)")
for it in m.relevant("what is the average discount?", "store_sales.csv", []): print("  ", it["key"])

m2 = Memory(p)  # new object = simulates a new session
print("\nAfter restart, items still there:", len(m2.items))
print(m2.pretty())
print("\nForget:", m2.forget("summary style"))
print("Remaining:", len(Memory(p).items))
