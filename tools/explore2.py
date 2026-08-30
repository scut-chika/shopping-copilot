"""Measure the ceiling: how discriminative is the FULL disclosed constraint set?"""
import json, sys
from collections import defaultdict, Counter
sys.path.insert(0, ".")
from evaluator.local_evaluator import intent_card, coarse_category, classify_constraint, _flatten_values, _clean_constraint

products = {}
with open("data/catalog.jsonl", encoding="utf-8") as fh:
    for line in fh:
        p = json.loads(line)
        products[str(p["parent_asin"])] = p

cards, cat_of = {}, {}
for asin, p in products.items():
    cards[asin] = intent_card(p)
    cat_of[asin] = coarse_category([str(v) for v in (p.get("categories") or [])])

# index: constraint string -> set of asins whose CARD contains it
card_index = defaultdict(set)
for asin, c in cards.items():
    for v in c["hard_constraints"] + c["soft_preferences"]:
        card_index[v].add(asin)

# index: constraint string -> set of asins whose FLATTENED features/details contain it (looser)
loose_index = defaultdict(set)
for asin, p in products.items():
    vals = [*_flatten_values(p.get("features")), *_flatten_values(p.get("details"))]
    for v in vals:
        cv = _clean_constraint(v, 180)
        if cv:
            loose_index[cv].add(asin)

cat_index = defaultdict(set)
for asin, c in cat_of.items():
    cat_index[c].add(asin)

sess = [json.loads(l) for l in open("data/public_set.jsonl", encoding="utf-8")]

def report(name, counts):
    counts = sorted(counts)
    n = len(counts)
    e1 = sum(1 for c in counts if c == 1)
    e10 = sum(1 for c in counts if c <= 10)
    e50 = sum(1 for c in counts if c <= 50)
    print(f"  {name:38s} =1: {e1/n:5.1%}   <=10: {e10/n:5.1%}   <=50: {e50/n:5.1%}   median={counts[n//2]}")

print("Candidate-set size after intersecting evidence (all 4 constraints disclosed):")
for label, idx in (("card-exact", card_index), ("loose feature/detail match", loose_index)):
    full, withcat = [], []
    for s in sess:
        t = str(s["ground_truth"]["parent_asin"])
        cs = cards[t]["hard_constraints"] + cards[t]["soft_preferences"]
        sets = [idx[v] for v in cs if v in idx]
        inter = set.intersection(*sets) if sets else set(products)
        full.append(len(inter))
        withcat.append(len(inter & cat_index[cat_of[t]]))
    report(label, full)
    report(label + " + category", withcat)

# progressive: how does it narrow turn by turn (card-exact + category)
print("\nProgressive narrowing (card-exact + category), median candidate count:")
for k in range(0, 5):
    counts = []
    for s in sess:
        t = str(s["ground_truth"]["parent_asin"])
        cs = (cards[t]["hard_constraints"] + cards[t]["soft_preferences"])[:k]
        sets = [card_index[v] for v in cs if v in card_index]
        inter = set.intersection(*sets) if sets else set(products)
        counts.append(len(inter & cat_index[cat_of[t]]))
    counts.sort()
    n = len(counts)
    print(f"  {k} constraints: median={counts[n//2]:6d}  p75={counts[int(n*.75)]:6d}  <=10: {sum(1 for c in counts if c<=10)/n:5.1%}")

# Does the target ALWAYS survive the intersection? (sanity: it must)
bad = sum(1 for s in sess
          if str(s["ground_truth"]["parent_asin"]) not in
          set.intersection(*[card_index[v] for v in cards[str(s["ground_truth"]["parent_asin"])]["hard_constraints"]+cards[str(s["ground_truth"]["parent_asin"])]["soft_preferences"] if v in card_index]))
print(f"\nsanity: sessions where target NOT in its own intersection = {bad} (must be 0)")
