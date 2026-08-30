"""Validate the core hypothesis: how much do (coarse_category, first_constraint) narrow the catalog?"""
import json, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, ".")
from evaluator.local_evaluator import intent_card, coarse_category, classify_constraint

products = {}
with open("data/catalog.jsonl", encoding="utf-8") as fh:
    for line in fh:
        p = json.loads(line)
        products[str(p["parent_asin"])] = p

print(f"catalog: {len(products)} products")

# 1. coarse_category distribution
cats = Counter()
cat_index = defaultdict(list)
for asin, p in products.items():
    c = coarse_category([str(v) for v in (p.get("categories") or [])])
    cats[c] += 1
    cat_index[c].append(asin)
print(f"\ndistinct coarse_category values: {len(cats)}")
print("top 10:", cats.most_common(10))
sizes = sorted(cats.values())
print(f"category bucket size: median={sizes[len(sizes)//2]}, p90={sizes[int(len(sizes)*0.9)]}, max={max(sizes)}")

# 2. Precompute intent cards for ALL products
cards = {asin: intent_card(p) for asin, p in products.items()}

# 3. How unique is (coarse_category, hard_constraints[0])?
key_index = defaultdict(list)
for asin, p in products.items():
    c = coarse_category([str(v) for v in (p.get("categories") or [])])
    hc = cards[asin]["hard_constraints"]
    if hc:
        key_index[(c, hc[0])].append(asin)

sess = [json.loads(l) for l in open("data/public_set.jsonl", encoding="utf-8")]
print(f"\npublic sessions: {len(sess)}")
buckets = Counter()
for s in sess:
    t = str(s["ground_truth"]["parent_asin"])
    p = products[t]
    c = coarse_category([str(v) for v in (p.get("categories") or [])])
    hc = cards[t]["hard_constraints"]
    n = len(key_index.get((c, hc[0]), [])) if hc else -1
    buckets[min(n, 100) if n > 0 else n] += 1

exact1 = sum(v for k, v in buckets.items() if k == 1)
le10 = sum(v for k, v in buckets.items() if 1 <= k <= 10)
le50 = sum(v for k, v in buckets.items() if 1 <= k <= 50)
print(f"sessions where (category, hard_constraint[0]) yields:")
print(f"   exactly 1 candidate : {exact1}/{len(sess)} = {exact1/len(sess):.1%}")
print(f"   <= 10 candidates    : {le10}/{len(sess)} = {le10/len(sess):.1%}")
print(f"   <= 50 candidates    : {le50}/{len(sess)} = {le50/len(sess):.1%}")

# 4. category-only narrowing
only_cat = []
for s in sess:
    t = str(s["ground_truth"]["parent_asin"])
    p = products[t]
    c = coarse_category([str(v) for v in (p.get("categories") or [])])
    only_cat.append(len(cat_index[c]))
only_cat.sort()
print(f"\ncategory-only candidate count: median={only_cat[len(only_cat)//2]}, p90={only_cat[int(len(only_cat)*0.9)]}")

# 5. how many constraints does a card actually carry
nconstraints = Counter(len(c["hard_constraints"]) + len(c["soft_preferences"]) for c in cards.values())
print(f"\nconstraints per card: {sorted(nconstraints.items())}")
attrs = Counter()
for s in sess:
    t = str(s["ground_truth"]["parent_asin"])
    for v in cards[t]["hard_constraints"] + cards[t]["soft_preferences"]:
        attrs[classify_constraint(v)] += 1
print(f"attribute distribution over session targets: {attrs.most_common()}")
