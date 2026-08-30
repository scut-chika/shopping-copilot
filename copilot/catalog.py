"""Catalog loading and the inverted indexes every retrieval route reads from.

All indexes are built once at process start and held in memory, per the
challenge constraint that no external vector-database cluster may be used.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from .config import Config
from .simulator_model import (
    card_constraints,
    classify_constraint,
    clean_constraint,
    coarse_category,
    flatten_values,
    searchable_text,
)

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

STOPWORDS = frozenset(
    """a an and are as at be but by for from i in is it me my of on or please some
    that the this to want with would you looking still exploring key requirement
    have preference additional judgment matters those options not quite right yet
    ask about one specific attribute actually ignore earlier need what""".split()
)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 1 and t.lower() not in STOPWORDS]


class CatalogIndex:
    """Immutable, read-only view over the frozen 50k-product catalog."""

    def __init__(self, catalog_path: str | Path, config: Config) -> None:
        self.config = config
        self.path = Path(catalog_path)

        self.products: dict[str, dict] = {}
        self.cards: dict[str, list[str]] = {}
        self.category_of: dict[str, str] = {}
        self.prior: dict[str, float] = {}

        self.card_index: dict[str, list[str]] = defaultdict(list)
        self.loose_index: dict[str, list[str]] = defaultdict(list)
        self.category_index: dict[str, list[str]] = defaultdict(list)

        self._attribute_cache: dict[str, str] = {}
        self._profile_text: dict[str, str] = {}

        self._load()
        self._build_fts()
        self._build_constraint_fts()

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                asin = str(product["parent_asin"])
                self.products[asin] = product

                constraints = card_constraints(product)
                self.cards[asin] = constraints
                for value in constraints:
                    self.card_index[value].append(asin)

                for raw in (
                    *flatten_values(product.get("features")),
                    *flatten_values(product.get("details")),
                ):
                    cleaned = clean_constraint(raw)
                    if cleaned:
                        self.loose_index[cleaned].append(asin)

                category = coarse_category([str(v) for v in (product.get("categories") or [])])
                self.category_of[asin] = category
                self.category_index[category].append(asin)

                self.prior[asin] = self._prior_for(product)
                self._profile_text[asin] = " ".join(
                    tokenize(f"{product.get('title') or ''} {' '.join(flatten_values(product.get('features')))}")
                )

        # freeze the defaultdicts so a missing key cannot silently grow them
        self.card_index = dict(self.card_index)
        self.loose_index = dict(self.loose_index)
        self.category_index = dict(self.category_index)

    @staticmethod
    def _prior_for(product: dict) -> float:
        """Weak popularity/quality prior, used only to break ties inside a tier."""
        try:
            rating = float(product.get("average_rating") or 0.0)
        except (TypeError, ValueError):
            rating = 0.0
        try:
            count = float(product.get("rating_number") or 0.0)
        except (TypeError, ValueError):
            count = 0.0
        return (rating / 5.0) * math.log1p(max(count, 0.0)) / 12.0

    # ------------------------------------------------------------------- fts

    def _build_fts(self) -> None:
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )

        def field(value: object) -> str:
            if value is None:
                return ""
            if isinstance(value, dict):
                return " ".join(f"{k} {v}" for k, v in value.items())
            if isinstance(value, list):
                return " ".join(str(v) for v in value)
            return str(value)

        batch: list[tuple] = []
        for asin, product in self.products.items():
            batch.append(
                (
                    asin,
                    field(product.get("title")),
                    field(product.get("categories")),
                    field(product.get("features")),
                    field(product.get("details")),
                    field(product.get("store")),
                    field(product.get("description")),
                )
            )
            if len(batch) >= 2000:
                cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
                batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
        self.connection.commit()

    def _build_constraint_fts(self) -> None:
        """A second FTS index, over the 60k distinct constraint strings.

        Used to recover what the customer said when their phrasing does not
        match a known template.
        """
        self.constraint_connection = sqlite3.connect(":memory:", check_same_thread=False)
        cursor = self.constraint_connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE cons USING fts5(text, tokenize='unicode61 remove_diacritics 2')"
        )
        batch = [(text,) for text in self.card_index]
        for start in range(0, len(batch), 5000):
            cursor.executemany("INSERT INTO cons VALUES (?)", batch[start:start + 5000])
        self.constraint_connection.commit()

    def mine_constraints(
        self,
        message: str,
        min_overlap: float,
        limit: int,
        max_results: int,
        min_tokens: int = 2,
    ) -> list[str]:
        """Constraints whose content tokens are mostly present in `message`.

        Template-independent, so it survives rewording; the overlap threshold
        keeps it from inventing constraints out of incidental word matches.
        """
        message_tokens = set(tokenize(message))
        if len(message_tokens) < 2:
            return []
        terms = list(message_tokens)[:48]
        expression = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = self.constraint_connection.execute(
                "SELECT text FROM cons WHERE cons MATCH ? ORDER BY bm25(cons) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.Error:
            return []
        scored: list[tuple[int, float, str]] = []
        for (text,) in rows:
            tokens = set(tokenize(text))
            if len(tokens) < max(min_tokens, 2):
                continue
            matched = len(tokens & message_tokens)
            overlap = matched / len(tokens)
            if overlap >= min_overlap:
                # Rank by how much was actually matched, not by ratio: a long
                # specific constraint carries more evidence than a short generic
                # one that happens to match completely.
                scored.append((matched, overlap, str(text)))
        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [text for _, _, text in scored[:max_results]]

    # --------------------------------------------------------------- queries

    def attribute_of(self, constraint: str) -> str:
        cached = self._attribute_cache.get(constraint)
        if cached is None:
            cached = classify_constraint(constraint)
            self._attribute_cache[constraint] = cached
        return cached

    def postings(self, constraint: str) -> list[str] | None:
        """Products whose reconstructed intent card contains this exact string."""
        return self.card_index.get(constraint)

    def loose_postings(self, constraint: str) -> list[str] | None:
        return self.loose_index.get(constraint)

    def in_category(self, category: str) -> list[str]:
        return self.category_index.get(category, [])

    def bm25(self, text: str, limit: int) -> list[str]:
        terms = list(dict.fromkeys(tokenize(text)))[:48]
        if not terms:
            return []
        expression = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [str(row[0]) for row in rows]

    def profile_overlap(self, asin: str, tags: list[str]) -> float:
        if not tags:
            return 0.0
        text = self._profile_text.get(asin, "")
        if not text:
            return 0.0
        hits = sum(1 for tag in tags if tag and tag.lower() in text)
        return hits / len(tags)

    def searchable(self, asin: str) -> str:
        return searchable_text(self.products[asin])

    def __len__(self) -> int:
        return len(self.products)
