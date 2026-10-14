"""Core word sense disambiguation (WSD) logic built on WordNet 3.0.

Provides:
  * sense inventory for a word (all synsets with glosses and lexical relations)
  * extended Lesk WSD that uses glosses, examples and WordNet lexical relations
  * most-frequent-sense (MFS) baseline
  * helpers that turn a text into a table of per-word sense predictions
"""
from __future__ import annotations

import math
import os
import warnings
from functools import lru_cache

import nltk
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# WordNet data is stored locally in ./nltk_data so the app is self-contained.
_HERE = os.path.dirname(os.path.abspath(__file__))
nltk.data.path.insert(0, os.path.join(_HERE, "nltk_data"))

from nltk.corpus import wordnet as wn  # noqa: E402

SPACY_TO_WN = {"NOUN": wn.NOUN, "PROPN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}
POS_NAMES = {wn.NOUN: "noun", wn.VERB: "verb", wn.ADJ: "adjective", wn.ADJ_SAT: "adjective", wn.ADV: "adverb"}

# Relations used to enrich a sense's signature in extended Lesk.
RELATION_FUNCS = {
    "hypernyms": lambda s: s.hypernyms() + s.instance_hypernyms(),
    "hyponyms": lambda s: s.hyponyms() + s.instance_hyponyms(),
    "part_meronyms": lambda s: s.part_meronyms(),
    "member_meronyms": lambda s: s.member_meronyms(),
    "substance_meronyms": lambda s: s.substance_meronyms(),
    "part_holonyms": lambda s: s.part_holonyms(),
    "member_holonyms": lambda s: s.member_holonyms(),
    "substance_holonyms": lambda s: s.substance_holonyms(),
    "similar_tos": lambda s: s.similar_tos(),
    "also_sees": lambda s: s.also_sees(),
    "attributes": lambda s: s.attributes(),
    "entailments": lambda s: s.entailments(),
    "causes": lambda s: s.causes(),
    "verb_groups": lambda s: s.verb_groups(),
}


# ----------------------------------------------------------------------------
# NLP pipeline (spaCy)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_nlp():
    import spacy

    return spacy.load("en_core_web_sm", disable=["ner"])


def content_lemmas(text: str, exclude_idx: set[int] | None = None, doc=None) -> list[str]:
    """Lower-cased lemmas of content words (nouns, verbs, adjectives, adverbs)."""
    doc = doc or get_nlp()(text)
    exclude_idx = exclude_idx or set()
    out = []
    for tok in doc:
        if tok.i in exclude_idx:
            continue
        if tok.pos_ in SPACY_TO_WN and not tok.is_stop and tok.is_alpha and len(tok.lemma_) > 1:
            out.append(tok.lemma_.lower())
    return out


# ----------------------------------------------------------------------------
# Sense inventory and lexical relations
# ----------------------------------------------------------------------------
def sense_frequency(synset, word: str) -> int:
    """SemCor-derived frequency count of `word` in this synset (WordNet cntlist)."""
    total = 0
    for lem in synset.lemmas():
        if lem.name().lower() == word.lower().replace(" ", "_"):
            total += lem.count()
    return total


def relations_of(synset) -> dict[str, list[str]]:
    """All non-empty lexical relations of a synset as {relation: [synset names]}."""
    rels = {}
    for name, fn in RELATION_FUNCS.items():
        try:
            vals = fn(synset)
        except Exception:
            vals = []
        if vals:
            rels[name] = [s.name() for s in vals]
    ants = [a.name() for lem in synset.lemmas() for a in lem.antonyms()]
    if ants:
        rels["antonyms"] = ants
    derived = sorted({d.name() for lem in synset.lemmas() for d in lem.derivationally_related_forms()})
    if derived:
        rels["derivationally_related"] = derived
    return rels


def sense_inventory(word: str, pos: str | None = None) -> list[dict]:
    """Every sense of `word` in WordNet, with gloss, examples, and relations."""
    rows = []
    for rank, syn in enumerate(wn.synsets(word.replace(" ", "_"), pos=pos), start=1):
        rels = relations_of(syn)
        rows.append(
            {
                "word": word,
                "rank": rank,
                "synset": syn.name(),
                "pos": POS_NAMES.get(syn.pos(), syn.pos()),
                "definition": syn.definition(),
                "examples": " | ".join(syn.examples()),
                "synonyms": ", ".join(l.name().replace("_", " ") for l in syn.lemmas()),
                "hypernyms": ", ".join(rels.get("hypernyms", [])),
                "hyponyms": ", ".join(rels.get("hyponyms", [])[:8]),
                "meronyms/holonyms": ", ".join(
                    f"{k}: {','.join(v)}" for k, v in rels.items() if "meronyms" in k or "holonyms" in k
                ),
                "antonyms": ", ".join(rels.get("antonyms", [])),
                "lexname": syn.lexname(),
                "sense_count": sense_frequency(syn, word),
            }
        )
    return rows


def inventory_frame(words: list[str]) -> pd.DataFrame:
    rows = []
    for w in words:
        rows.extend(sense_inventory(w))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Extended Lesk
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _gloss_bag(synset_name: str) -> tuple[str, ...]:
    syn = wn.synset(synset_name)
    text = " ".join([syn.definition()] + syn.examples())
    return tuple(content_lemmas(text))


@lru_cache(maxsize=None)
def signature(synset_name: str) -> tuple[dict, dict]:
    """(own_bag, related_bag) as {lemma: weight} dictionaries.

    own_bag: gloss + examples + synonyms of the sense itself (weight 2).
    related_bag: glosses/synonyms of related synsets from the lexical relations (weight 1).
    """
    syn = wn.synset(synset_name)
    own: dict[str, float] = {}
    for w in _gloss_bag(synset_name):
        own[w] = 2.0
    for lem in syn.lemmas():
        for part in lem.name().lower().split("_"):
            own.setdefault(part, 2.0)
    related: dict[str, float] = {}
    for name, fn in RELATION_FUNCS.items():
        try:
            rel_syns = fn(syn)
        except Exception:
            rel_syns = []
        for r in rel_syns:
            for w in _gloss_bag(r.name()):
                related.setdefault(w, 1.0)
            for lem in r.lemmas():
                for part in lem.name().lower().split("_"):
                    related.setdefault(part, 1.0)
    return own, related


def lesk_scores(context: list[str], synsets, word: str) -> list[tuple[float, object]]:
    ctx = set(context)
    out = []
    for syn in synsets:
        own, related = signature(syn.name())
        score = sum(wt for w, wt in own.items() if w in ctx)
        score += sum(wt for w, wt in related.items() if w in ctx and w not in own)
        out.append((score, syn))
    return out


def disambiguate_word(
    word: str,
    context: list[str],
    pos: str | None = None,
    method: str = "lesk",
    proper: bool = False,
):
    """Return (synset | None, score, n_candidates, method_used).

    method = 'lesk'  -> extended Lesk with relation-enriched signatures; falls back to MFS on zero overlap
    method = 'mfs'   -> most frequent sense baseline (first WordNet sense)
    """
    cands = wn.synsets(word.replace(" ", "_"), pos=pos)
    if not proper:
        # Instance synsets (named people, places, e.g. crane.n.01 = the writer
        # Stephen Crane) are only plausible for proper nouns.
        common = [c for c in cands if not c.instance_hypernyms()]
        cands = common or cands
    if not cands:
        return None, 0.0, 0, "none"
    if len(cands) == 1:
        return cands[0], 1.0, 1, "monosemous"
    if method == "mfs":
        return cands[0], 0.0, len(cands), "mfs"
    scored = lesk_scores(context, cands, word)
    best_score = max(s for s, _ in scored)
    if best_score <= 0:
        return cands[0], 0.0, len(cands), "mfs-fallback"
    # Among ties prefer the sense with the higher corpus frequency / WordNet order.
    tied = [(sense_frequency(s, word), -i, s) for i, (sc, s) in enumerate(scored) if sc == best_score]
    tied.sort(reverse=True)
    return tied[0][2], float(best_score), len(cands), "lesk"


def disambiguate_text(text: str, method: str = "lesk", window: int | None = None, only_ambiguous: bool = False) -> pd.DataFrame:
    """Disambiguate every content word in `text`; one output row per word."""
    nlp = get_nlp()
    doc = nlp(text)
    rows = []
    for si, sent in enumerate(doc.sents, start=1):
        sent_tokens = [t for t in sent]
        for tok in sent_tokens:
            if tok.pos_ not in SPACY_TO_WN or not tok.is_alpha or tok.is_stop:
                continue
            wpos = SPACY_TO_WN[tok.pos_]
            lemma = tok.lemma_.lower()
            if window:
                lo, hi = tok.i - window, tok.i + window
                ctx_doc_tokens = [t for t in doc if lo <= t.i <= hi and t.i != tok.i]
            else:
                ctx_doc_tokens = [t for t in sent_tokens if t.i != tok.i]
            ctx = [
                t.lemma_.lower()
                for t in ctx_doc_tokens
                if t.pos_ in SPACY_TO_WN and not t.is_stop and t.is_alpha and len(t.lemma_) > 1
            ]
            syn, score, n, used = disambiguate_word(
                lemma, ctx, pos=wpos, method=method, proper=(tok.pos_ == "PROPN")
            )
            if syn is None:
                continue
            if only_ambiguous and n < 2:
                continue
            rows.append(
                {
                    "sentence": si,
                    "tok_i": tok.i,
                    "token": tok.text,
                    "lemma": lemma,
                    "pos": POS_NAMES[wpos],
                    "predicted_synset": syn.name(),
                    "definition": syn.definition(),
                    "n_senses": n,
                    "score": score,
                    "method": used,
                    "context_words": ", ".join(ctx),
                    "proper": tok.pos_ == "PROPN",
                    "lexname": syn.lexname(),
                }
            )
    return pd.DataFrame(rows)


def candidate_breakdown(word: str, context: list[str], pos: str | None = None, proper: bool = False, method: str = "lesk"):
    """Explain a decision: every candidate sense with its score and the context
    words that supported it. Returns (rows, method_used)."""
    cands = wn.synsets(word.replace(" ", "_"), pos=pos)
    if not proper:
        cands = [c for c in cands if not c.instance_hypernyms()] or cands
    chosen, _, _, used = disambiguate_word(word, context, pos=pos, method=method, proper=proper)
    ctx = set(context)
    rows = []
    for syn in cands:
        own, rel = signature(syn.name())
        own_hits = sorted(x for x in own if x in ctx)
        rel_hits = sorted(x for x in rel if x in ctx and x not in own)
        rows.append(
            {
                "synset": syn.name(),
                "definition": syn.definition(),
                "examples": syn.examples(),
                "lexname": syn.lexname(),
                "score": 2.0 * len(own_hits) + 1.0 * len(rel_hits),
                "own_hits": own_hits,
                "rel_hits": rel_hits,
                "freq": sense_frequency(syn, word),
                "chosen": chosen is not None and syn.name() == chosen.name(),
            }
        )
    return rows, used


def relation_graph_dot(synset_name: str, show: set[str], max_per_relation: int = 6) -> str:
    """Graphviz DOT source: the sense in the centre, selected relations around it."""
    syn = wn.synset(synset_name)
    short = lambda s: s.name().split(".")[0].replace("_", " ") + "." + ".".join(s.name().split(".")[1:])
    lines = [
        "digraph G {",
        "rankdir=BT; bgcolor=transparent; pad=0.2; nodesep=0.35; ranksep=0.5;",
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#4A537F", fillcolor="#222845", fontcolor="#EEF0FA", margin="0.14,0.07"];',
        'edge [fontname="Helvetica", fontsize=9, color="#8C9BFF", fontcolor="#9EA5CC"];',
        f'"{syn.name()}" [label="{short(syn)}", fillcolor="#F5C84C", fontcolor="#241B00", color="#F5C84C", penwidth=1.6];',
    ]
    seen = {syn.name()}

    def add(other, label, src, dst):
        if other.name() not in seen:
            seen.add(other.name())
            lines.append(f'"{other.name()}" [label="{short(other)}"];')
        lines.append(f'"{src}" -> "{dst}" [label="{label}"];')

    if "hypernym path" in show:
        path = syn.hypernym_paths()[0] if syn.hypernym_paths() else []
        for child, parent in zip(reversed(path[1:]), reversed(path[:-1])):
            for n in (child, parent):
                if n.name() not in seen:
                    seen.add(n.name())
                    lines.append(f'"{n.name()}" [label="{short(n)}"];')
            lines.append(f'"{child.name()}" -> "{parent.name()}" [label="is a"];')
    spec = [
        ("hyponyms", "is a", lambda s: s.hyponyms(), "child"),
        ("part meronyms", "part of", lambda s: s.part_meronyms(), "child"),
        ("member meronyms", "member of", lambda s: s.member_meronyms(), "child"),
        ("holonyms", "has part", lambda s: s.part_holonyms() + s.member_holonyms() + s.substance_holonyms(), "parent"),
        ("similar to", "similar", lambda s: s.similar_tos(), "parent"),
    ]
    for key, label, fn, side in spec:
        if key in show:
            for other in fn(syn)[:max_per_relation]:
                if side == "child":
                    add(other, label, other.name(), syn.name())
                else:
                    add(other, label, syn.name(), other.name())
    if "antonyms" in show:
        for lem in syn.lemmas():
            for ant in lem.antonyms()[:3]:
                a = ant.synset()
                if a.name() not in seen:
                    seen.add(a.name())
                    lines.append(f'"{a.name()}" [label="{short(a)}", fillcolor="#4A2A38", color="#C96A85"];')
                lines.append(f'"{syn.name()}" -> "{a.name()}" [label="opposite", style=dashed, dir=both];')
    lines.append("}")
    return "\n".join(lines)


def evaluate(gold: list[dict], method: str) -> dict:
    """gold items: {sentence, target, pos, gold_synset}. Returns accuracy + details."""
    nlp = get_nlp()
    correct = 0
    details = []
    for item in gold:
        doc = nlp(item["sentence"])
        target = item["target"].lower()
        tok = next((t for t in doc if t.text.lower() == target or t.lemma_.lower() == target), None)
        ctx = content_lemmas(item["sentence"], exclude_idx={tok.i} if tok else None, doc=doc)
        syn, score, n, used = disambiguate_word(
            tok.lemma_.lower() if tok else target,
            ctx,
            pos=item.get("pos"),
            method=method,
            proper=bool(tok and tok.pos_ == "PROPN"),
        )
        pred = syn.name() if syn else None
        ok = pred == item["gold_synset"]
        correct += ok
        details.append({**item, "predicted": pred, "correct": ok, "score": score, "used": used})
    return {"method": method, "accuracy": correct / len(gold), "n": len(gold), "details": details}
