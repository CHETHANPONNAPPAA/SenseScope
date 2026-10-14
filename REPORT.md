# SenseScope: Word Sense Disambiguation with WordNet. Short Analysis Report

## 1. What was built
SenseScope is a Python package and Streamlit app that (a) lists every WordNet sense of a word with its lexical relations, and (b) disambiguates words in running text.

| Component | File |
|---|---|
| Sense inventory, relations, extended Lesk, MFS baseline | `wsd_core.py` |
| Web app (sense explorer, WSD on any text, evaluation) | `app.py` |
| Test set (30 sentences) and demo text | `gold_data.py` |
| Script that regenerates all deliverables | `run_demo.py` → `outputs/` |

Tools: WordNet 3.0 via NLTK's corpus reader, spaCy (`en_core_web_sm`) for sentence splitting, POS tags and lemmas, pandas for tables.

## 2. Method
* **Sense inventory.** For a word, every synset is listed with gloss, examples, synonyms, hypernyms, hyponyms, meronyms/holonyms, antonyms, lexical file and its SemCor frequency count.
* **Extended Lesk.** Each candidate sense gets a signature: lemmas from its own gloss, examples and synonyms (weight 2) plus lemmas from the glosses and synonyms of related synsets reached through WordNet relations (hypernym, hyponym, meronym, holonym, similar-to, also-see, attribute, entailment, cause, verb-group; weight 1). The sense whose signature overlaps most with the sentence's content-word lemmas wins. Ties go to the sense with the higher corpus frequency. If nothing overlaps, the system falls back to the most frequent sense (`mfs-fallback`).
* **Filtering.** Instance synsets (e.g. `crane.n.01`, the writer Stephen Crane) are excluded unless spaCy tags the token as a proper noun.
* **Baseline.** Most frequent sense (first WordNet sense for the lemma and POS).

## 3. Results
**Sense inventory.** 146 senses for 10 target words, in `outputs/sense_inventory.csv`. Counts: light 47 (all parts of speech), match 19, bank 18, spring 11, bat 10, interest 10, plant 10, bass 9, crane 6, mouse 6. This shows how fine-grained WordNet is.

**Sample text** (`outputs/wsd_results.csv`, 10 sentences). By my own reading, 9 of 10 target words received a sensible sense: both *bank* senses, both *plant* senses, both *crane* senses, *mouse* (device), *mouse* (rodent) and *bass* (guitar). One error: "caught a huge bass in the lake" was labelled `sea_bass.n.01` (the fish *as food*) instead of `bass.n.08` (the fish). The rodent *mouse* was right only because of the MFS fallback; the context had no overlapping words.

**Test set** (30 sentences, 10 nouns, 3 each, `outputs/evaluation.csv`):

| Method | Accuracy |
|---|---|
| Most frequent sense | 43.3% (13/30) |
| Extended Lesk | 83.3% (25/30) |

Lesk fell back to MFS on 2 sentences.

## 4. Error analysis (all 5 Lesk errors)
| Sentence (target) | Predicted | Gold | Cause |
|---|---|---|---|
| "turned up the **bass**…deeper sound" | `bass.n.02` | `bass.n.01` | Two near-identical music senses; WordNet is very fine-grained. |
| "caught a large **bass**…grilled the fish" | `sea_bass.n.01` | `bass.n.08` | The fish-as-food sense matches "fish"/"grill"; animal vs. food regular polysemy. |
| "swung the **bat**…hit the baseball" | `bat.n.02` | `bat.n.05` | Exact tie (4.0 vs 4.0). The frequency tie-break picked the baseball "turn at bat" sense. |
| "garden blooms in **spring**…cold winter" | `spring.n.02` | `spring.n.01` | The season sense had zero overlap; one stray word ("long") pushed the metal-spring sense ahead. |
| "**crane** stood on one leg…wings" | `crane.n.04` | `crane.n.05` | No overlap at all (fallback). The bird sense only links to "wings" two hypernym levels up (wading bird → bird), but only one hop is used. |

## 5. Caveats
* The test set is small and written by the project author with context cues in each sentence, so it likely flatters Lesk. The MFS baseline is low because the set deliberately targets less frequent senses. Neither number should be read as a benchmark score. A proper evaluation would use SemCor or the Senseval/SemEval all-words datasets, which could not be downloaded in the build environment.
* The instance-synset filter was added after the first error review on this same set. It did not change Lesk's accuracy (it did raise the MFS baseline from 40.0% to 43.3% by removing proper-noun senses).
* Word-overlap methods ignore word order and syntax, so they are sensitive to a single stray overlap.

## 6. Next steps
1. Traverse the full hypernym chain (not just one hop) when building signatures.
2. Weight overlap words by inverse document frequency so rare words count more than "long" or "hit".
3. Add a supervised model with scikit-learn (e.g. logistic regression on context bag-of-words and neighbouring-word features) trained on SemCor, then compare it with Lesk.
4. Evaluate on SemCor / Senseval-style data and report per-POS accuracy.
