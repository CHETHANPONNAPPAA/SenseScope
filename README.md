# SenseScope

See which meaning a word carries in a sentence, and why. SenseScope runs word sense disambiguation (extended Lesk) on WordNet 3.0 and shows the evidence behind every decision.

## What's in the app
* **Word lab**: click an example or type a sentence, pick an ambiguous word, and see the chosen sense, a score chart for every candidate sense, and the context words that supported each one.
* **Annotate a text**: paste a paragraph; words are highlighted by kind of meaning, hover for the chosen sense, click a word to inspect it.
* **Browse senses**: look up any word, filter senses by definition text, and draw an interactive graph of its WordNet relations.
* **How well does it work?**: runs Lesk and the frequency baseline on 30 labelled sentences.

The header switches between the two methods and sets the context window.

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
WordNet 3.0 is bundled in `nltk_data/corpora/wordnet`, so no NLTK download is needed.

## Run
```bash
streamlit run app.py      # interactive app
python run_demo.py        # regenerates outputs/*.csv
```

## Outputs (`outputs/`)
* `sense_inventory.csv`: all senses and relations for 10 ambiguous words
* `wsd_results.csv`: WSD predictions for the demo text
* `evaluation.csv`, `evaluation_summary.json`: Lesk vs. most-frequent-sense on 30 labelled sentences

See `REPORT.md` for the analysis.
