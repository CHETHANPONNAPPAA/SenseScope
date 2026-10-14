"""SenseScope: see which meaning a word carries, and why.

Run with:  streamlit run app.py
"""
import html

import altair as alt
import pandas as pd
import streamlit as st
from nltk.corpus import wordnet as wn

import wsd_core as w
from gold_data import DEMO_TEXT, GOLD_ITEMS

APP_NAME = "SenseScope"
INK, MARKER, MUTED = "#8C9BFF", "#F5C84C", "#4A537F"

st.set_page_config(page_title=f"{APP_NAME} · word sense lab", page_icon="🔍", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700&family=DM+Sans:wght@400;500&display=swap');
html, body, [class*="st-"], .stMarkdown { font-family: 'DM Sans', system-ui, sans-serif; }
h1, h2, h3, .ss-wordmark, .ss-sense { font-family: 'Sora', 'DM Sans', sans-serif !important; letter-spacing: -0.015em; }
[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] * { font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important; }
.block-container { padding-top: 4.75rem; max-width: 1180px; }

.ss-wordmark { font-size: 2.7rem; font-weight: 700; line-height: 1.2; color: #EEF0FA; margin: 0; padding-top: .15rem; }
.ss-bar { width: 4.2rem; height: 6px; border-radius: 3px; background: #F5C84C; margin: .3rem 0 .85rem; }
.ss-tagline { color: #9EA5CC; font-size: 1.05rem; max-width: 40rem; line-height: 1.55; margin-bottom: 1.6rem; }

/* the answer: a marker swipe under the winning word */
.ss-answer { background: #181C30; border: 1px solid #2B3152; border-radius: 14px; padding: 1.1rem 1.3rem; }
.ss-sense { font-size: 1.55rem; font-weight: 700; color: #EEF0FA; }
.ss-sense mark { background: linear-gradient(transparent 58%, rgba(245,200,76,.55) 58%, rgba(245,200,76,.55) 92%, transparent 92%); color: inherit; padding: 0 .1em; }
.ss-def { font-size: 1.05rem; margin: .45rem 0 .2rem; color: #EEF0FA; }
.ss-meta { color: #9EA5CC; font-size: .9rem; }
.ss-ex { color: #B6BCDD; font-style: italic; margin-top: .5rem; font-size: .95rem; }

/* evidence chips */
.ss-chip { display: inline-block; padding: .08rem .55rem; margin: .12rem .2rem .12rem 0; border-radius: 999px; font-size: .86rem; border: 1px solid #3A4270; background: #222845; color: #BFC8FF; }
.ss-chip.own { background: #F5C84C; border-color: #F5C84C; color: #241B00; }
.ss-none { color: #8189B3; font-size: .9rem; }

/* annotated text */
.ss-text { background: #181C30; border: 1px solid #2B3152; border-radius: 14px; padding: 1.2rem 1.4rem; line-height: 2.15; font-size: 1.08rem; max-width: 62rem; color: #EEF0FA; }
.ss-w { padding: .05rem .22rem; border-radius: 4px; border-bottom: 3px solid var(--c); background: color-mix(in srgb, var(--c) 22%, transparent); cursor: help; }
.ss-w.sel { background: #F5C84C; color: #241B00; box-shadow: 0 0 0 2px #8C9BFF; }
.ss-w.mono { border-bottom-style: dotted; background: transparent; }
.ss-legend span { display: inline-block; margin-right: .9rem; font-size: .88rem; color: #9EA5CC; }
.ss-legend i { display: inline-block; width: .8rem; height: .8rem; border-radius: 3px; margin-right: .3rem; vertical-align: -1px; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)

PALETTE = ["#6CB4FF", "#FF9F68", "#5FD3A0", "#FF7EB6", "#B49BFF", "#F2C14E", "#4FD6E6", "#FF8A8A", "#A8B3CF", "#8C9BFF"]


def lex_color(lexname: str) -> str:
    return PALETTE[sum(map(ord, lexname)) % len(PALETTE)]


# ------------------------------------------------------------------ cached helpers
@st.cache_data(show_spinner=False)
def cached_inventory(word: str, pos: str | None) -> pd.DataFrame:
    return pd.DataFrame(w.sense_inventory(word, pos))


@st.cache_data(show_spinner=False)
def cached_wsd(text: str, method: str, window: int, only_amb: bool) -> pd.DataFrame:
    return w.disambiguate_text(text, method=method, window=window or None, only_ambiguous=only_amb)


@st.cache_data(show_spinner=False)
def cached_eval(method: str):
    return w.evaluate(GOLD_ITEMS, method)


def chips(words: list[str], own: bool) -> str:
    if not words:
        return '<span class="ss-none">no matching words</span>'
    cls = "ss-chip own" if own else "ss-chip"
    return "".join(f'<span class="{cls}">{html.escape(x)}</span>' for x in words)


METHOD_LABELS = {"lesk": "Context + relations", "mfs": "Most frequent sense"}


# ------------------------------------------------------------------ the inspector
def render_inspector(row: pd.Series, method: str, key: str):
    """Shows the chosen sense, a score chart for every candidate, and the evidence behind each."""
    ctx = [c for c in str(row["context_words"]).split(", ") if c]
    pos_code = {"noun": "n", "verb": "v", "adjective": "a", "adverb": "r"}[row["pos"]]
    rows, used = w.candidate_breakdown(row["lemma"], ctx, pos_code, bool(row["proper"]), method)
    chosen = next(r for r in rows if r["chosen"])
    ranked = sorted(rows, key=lambda r: -r["score"])
    margin = ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else ranked[0]["score"]

    left, right = st.columns([1.05, 1])
    with left:
        verdict = {
            "lesk": f"Chosen because the context shares {len(chosen['own_hits']) + len(chosen['rel_hits'])} word(s) with this sense.",
            "mfs-fallback": "No context word matched any sense, so the most common sense was used.",
            "mfs": "Baseline: always the most common sense, whatever the context says.",
            "monosemous": "Only one sense exists.",
        }.get(used, "")
        ex = f'<div class="ss-ex">“{html.escape(chosen["examples"][0])}”</div>' if chosen["examples"] else ""
        st.markdown(
            f"""<div class="ss-answer">
<div class="ss-sense"><mark>{html.escape(row['lemma'])}</mark></div>
<div class="ss-def">{html.escape(chosen['definition'])}</div>
<div class="ss-meta">{html.escape(chosen['synset'])} · {html.escape(chosen['lexname'])} · seen {chosen['freq']}× in the SemCor corpus</div>
{ex}
<div class="ss-meta" style="margin-top:.7rem">{verdict}</div>
</div>""",
            unsafe_allow_html=True,
        )
        if used == "lesk" and len(ranked) > 1:
            st.caption(f"Lead over the runner-up: {margin:g} point(s). A small lead means the call is a close one.")
    with right:
        df = pd.DataFrame(
            {
                "sense": [r["synset"] for r in rows],
                "score": [r["score"] for r in rows],
                "chosen": [r["chosen"] for r in rows],
                "definition": [r["definition"] for r in rows],
            }
        )
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusEnd=3, height=18)
            .encode(
                x=alt.X("score:Q", title="Overlap score", axis=alt.Axis(tickMinStep=1, grid=False)),
                y=alt.Y("sense:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False, labelFontSize=12)),
                color=alt.condition(alt.datum.chosen, alt.value("#F5C84C"), alt.value(MUTED)),
                tooltip=["sense", "score", "definition"],
            )
            .properties(height=alt.Step(26), title="How well each sense fits")
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, width="stretch")

    st.markdown("**Evidence for each sense**")
    st.caption("Yellow chips are words from the sense's own definition and examples (worth 2 points); blue chips come from related senses such as its parent or parts (1 point).")
    for r in ranked:
        tag = "chosen" if r["chosen"] else f"score {r['score']:g}"
        with st.expander(f"{r['synset']}: {r['definition'][:70]}{'…' if len(r['definition']) > 70 else ''}  ({tag})", expanded=r["chosen"]):
            st.markdown(chips(r["own_hits"], True) + chips(r["rel_hits"], False), unsafe_allow_html=True)

    with st.expander("Where this sense sits in WordNet"):
        rel_options = ["hypernym path", "hyponyms", "part meronyms", "member meronyms", "holonyms", "similar to", "antonyms"]
        shown = st.pills("Show", rel_options, selection_mode="multi", default=["hypernym path", "hyponyms"], key=f"{key}_rels")
        st.graphviz_chart(w.relation_graph_dot(chosen["synset"], set(shown or ["hypernym path"])), width="content")


# ------------------------------------------------------------------ header
head_l, head_r = st.columns([3.3, 2], vertical_alignment="bottom")
with head_l:
    st.markdown('<div class="ss-wordmark">SenseScope</div><div class="ss-bar"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ss-tagline">Words mean different things in different sentences. Pick a word, and SenseScope shows which WordNet meaning fits the context and which clues decided it.</div>',
        unsafe_allow_html=True,
    )
with head_r:
    method = st.segmented_control(
        "How to choose a sense", list(METHOD_LABELS), format_func=METHOD_LABELS.get, default="lesk", key="method",
        help="Context + relations compares the sentence with each sense's definition and related senses. Most frequent sense ignores the sentence and always picks the commonest meaning.",
    ) or "lesk"
    with st.popover("Context window"):
        window = st.slider("Words of context on each side", 0, 15, 0, key="window", help="0 uses the whole sentence.")

tab_lab, tab_text, tab_words, tab_eval = st.tabs(["Word lab", "Annotate a text", "Browse senses", "How well does it work?"])

# ------------------------------------------------------------------ Word lab
EXAMPLES = {
    "bank: money": "She deposited her paycheck at the bank on Friday.",
    "bank: river": "We had a picnic on the grassy bank of the river and watched the water.",
    "mouse: computer": "Click the left button of the mouse to select the file on the computer screen.",
    "mouse: animal": "The cat chased a small mouse, a rodent with a long tail, across the kitchen.",
    "bat: animal": "A bat flew out of the cave at night, hunting insects with its wings.",
    "crane: machine": "The construction crane lifted a heavy steel beam to the top of the building.",
    "spring: water": "Fresh water flows from the spring in the mountain valley.",
}


def load_example():
    pick = st.session_state.get("example_pick")
    if pick:
        st.session_state["lab_sentence"] = EXAMPLES[pick]
        st.session_state["lab_target"] = pick.split(":")[0]  # the word the example is about


def clear_target():
    st.session_state["lab_target"] = None  # user typed their own sentence


with tab_lab:
    st.pills("Try an example", list(EXAMPLES), key="example_pick", on_change=load_example)
    st.session_state.setdefault("lab_sentence", EXAMPLES["bank: money"])
    st.session_state.setdefault("lab_target", "bank")
    sentence = st.text_input("Or write your own sentence", key="lab_sentence", on_change=clear_target)
    if sentence.strip():
        df = cached_wsd(sentence, method, window, True)
        if df.empty:
            st.info("No ambiguous words found. Try a sentence with a word like bank, plant or light.")
        else:
            labels = {i: f"{r.token} ({r.n_senses} senses)" for i, r in df.iterrows()}

            def default_idx():
                target = st.session_state.get("lab_target")
                hit = df[df["lemma"] == target] if target else df.iloc[0:0]
                return hit.index[0] if not hit.empty else df["n_senses"].idxmax()

            # a new sentence (or a stale selection) resets the pick to the most interesting word
            if st.session_state.get("lab_sig") != sentence or st.session_state.get("lab_pick") not in labels:
                st.session_state["lab_pick"] = default_idx()
                st.session_state["lab_sig"] = sentence
            st.markdown("**Pick a word to inspect**")
            idx = st.pills("Ambiguous words", list(labels), format_func=labels.get, key="lab_pick", label_visibility="collapsed")
            if idx is None:
                idx = default_idx()
            render_inspector(df.loc[idx], method, key="lab")

            if method == "lesk":
                base = cached_wsd(sentence, "mfs", window, True).loc[idx]
                same = base["predicted_synset"] == df.loc[idx, "predicted_synset"]
                st.info(
                    f"The most-frequent-sense baseline would pick **{base['predicted_synset']}**: {base['definition']}"
                    + (" Same answer, so this sentence doesn't need the context." if same else " Different answer, so the context made the difference.")
                )

# ------------------------------------------------------------------ Annotate a text
with tab_text:
    text = st.text_area("Paste any English text", value=DEMO_TEXT, height=150, key="wsd_text")
    only_amb = st.toggle("Highlight only words with more than one sense", value=True, key="only_amb")
    if text.strip():
        df = cached_wsd(text, method, window, only_amb)
        if df.empty:
            st.info("No content words with WordNet senses were found.")
        else:
            doc = w.get_nlp()(text)
            by_char = {r.tok_i: r for r in df.reset_index().itertuples()}
            picked = st.session_state.get("ann_idx")
            parts, used_lex = [], {}  # lexname -> color, filled in order of first appearance
            for tok in doc:
                esc = html.escape(tok.text_with_ws)
                r = by_char.get(tok.i)
                if r is None:
                    parts.append(esc)
                    continue
                c = used_lex.setdefault(r.lexname, PALETTE[len(used_lex) % len(PALETTE)])
                tip = html.escape(f"{r.predicted_synset}: {r.definition}", quote=True)
                cls = "ss-w" + (" sel" if picked == r.index else "") + (" mono" if r.n_senses < 2 else "")
                parts.append(f'<span class="{cls}" style="--c:{c}" title="{tip}">{html.escape(tok.text)}</span>{html.escape(tok.whitespace_)}')
            st.markdown(f'<div class="ss-text">{"".join(parts)}</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="ss-legend" style="margin-top:.6rem">'
                + "".join(f'<span><i style="background:{c}"></i>{html.escape(k)}</span>' for k, c in sorted(used_lex.items()))
                + "</div>",
                unsafe_allow_html=True,
            )
            st.caption("Hover a highlighted word to read its chosen meaning. Colors mark the kind of thing the meaning is about.")

            amb = df[df["n_senses"] > 1]
            if not amb.empty:
                labels = {i: f"{r.token} → {r.predicted_synset.split('.')[0]}" for i, r in amb.iterrows()}
                st.markdown("**Inspect a word**")
                sel = st.pills("Inspect", list(labels), format_func=labels.get, key="ann_idx", label_visibility="collapsed")
                if sel is not None:
                    render_inspector(df.loc[sel], method, key="ann")
            with st.expander("Table view and download"):
                st.dataframe(df[["sentence", "token", "pos", "predicted_synset", "definition", "n_senses", "score", "method"]], hide_index=True)
                st.download_button("Download results (CSV)", df.to_csv(index=False), "wsd_results.csv", "text/csv")

# ------------------------------------------------------------------ Browse senses
with tab_words:
    st.pills("Popular ambiguous words", ["bank", "bass", "bat", "plant", "spring", "mouse", "crane", "match", "interest", "light"],
             key="word_pick", on_change=lambda: st.session_state.update(inv_word=st.session_state["word_pick"] or st.session_state.get("inv_word", "bank")))
    st.session_state.setdefault("inv_word", "bank")
    c1, c2 = st.columns([3, 2])
    word = c1.text_input("Look up a word", key="inv_word").strip()
    pos_label = c2.segmented_control("Part of speech", ["any", "noun", "verb", "adjective", "adverb"], default="any", key="inv_pos")
    pos = {"any": None, "noun": "n", "verb": "v", "adjective": "a", "adverb": "r", None: None}[pos_label]
    if word:
        inv = cached_inventory(word, pos)
        if inv.empty:
            st.warning(f"WordNet has no entry for “{word}”. Check the spelling or try its base form (for example “run” instead of “running”).")
        else:
            st.markdown(f"#### {len(inv)} senses of “{word}”")
            q = st.text_input("Filter by a word in the definition", key="inv_filter", placeholder="for example: money, water, animal")
            view = inv[inv["definition"].str.contains(q, case=False, na=False)] if q else inv
            if view.empty:
                st.info("No sense matches that filter.")
            else:
                choice = st.selectbox("Open a sense", view["synset"].tolist(), format_func=lambda s: f"{s}: {wn.synset(s).definition()[:80]}", key="inv_syn")
                syn = wn.synset(choice)
                opts = ["hypernym path", "hyponyms", "part meronyms", "member meronyms", "holonyms", "similar to", "antonyms"]
                shown = st.pills("Relations to draw", opts, selection_mode="multi", default=["hypernym path", "hyponyms"], key="inv_rels")
                cA, cB = st.columns([1, 1.15])
                with cA:
                    ex = f'<div class="ss-ex">“{html.escape(syn.examples()[0])}”</div>' if syn.examples() else ""
                    st.markdown(
                        f"""<div class="ss-answer"><div class="ss-sense"><mark>{html.escape(syn.name())}</mark></div>
<div class="ss-def">{html.escape(syn.definition())}</div>
<div class="ss-meta">{html.escape(syn.lexname())} · also called: {html.escape(', '.join(l.name().replace('_', ' ') for l in syn.lemmas()))}</div>{ex}</div>""",
                        unsafe_allow_html=True,
                    )
                    rels = w.relations_of(syn)
                    if rels:
                        st.dataframe(pd.DataFrame({"relation": [k.replace("_", " ") for k in rels], "related senses": [", ".join(v[:6]) for v in rels.values()]}), hide_index=True)
                with cB:
                    st.graphviz_chart(w.relation_graph_dot(choice, set(shown or ["hypernym path"])), width="content")
                with st.expander("All senses as a table"):
                    st.dataframe(view[["rank", "synset", "pos", "definition", "synonyms", "lexname", "sense_count"]], hide_index=True)
                    st.download_button("Download sense inventory (CSV)", inv.to_csv(index=False), f"sense_inventory_{word}.csv", "text/csv")

# ------------------------------------------------------------------ Evaluation
with tab_eval:
    st.markdown(
        f"Thirty sentences with the right meaning labelled by hand, covering ten ambiguous nouns. This is a small sanity check, not a benchmark: the sentences were written for this project."
    )
    if st.button("Run the test", type="primary", key="eval_go"):
        with st.status("Disambiguating 30 sentences with both methods", expanded=False) as status:
            lesk, mfs = cached_eval("lesk"), cached_eval("mfs")
            status.update(label="Done", state="complete")
        st.session_state["eval"] = (lesk, mfs)
    if "eval" in st.session_state:
        lesk, mfs = st.session_state["eval"]
        m1, m2 = st.columns(2)
        m1.metric("Lesk (context + relations)", f"{lesk['accuracy']:.1%}", f"{(lesk['accuracy'] - mfs['accuracy']) * 100:+.1f} points vs baseline")
        m2.metric("Most frequent sense", f"{mfs['accuracy']:.1%}")
        rows = [
            {"target": a["target"], "sentence": a["sentence"], "right answer": a["gold_synset"], "Lesk": a["predicted"], "Lesk correct": a["correct"], "baseline": b["predicted"], "baseline correct": b["correct"]}
            for a, b in zip(lesk["details"], mfs["details"])
        ]
        ev = pd.DataFrame(rows)
        per_word = ev.groupby("target")[["Lesk correct", "baseline correct"]].mean().reset_index().melt("target", var_name="method", value_name="accuracy")
        chart = (
            alt.Chart(per_word)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("target:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
                xOffset="method:N",
                y=alt.Y("accuracy:Q", title="Share correct", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("method:N", scale=alt.Scale(domain=["Lesk correct", "baseline correct"], range=["#8C9BFF", MUTED]), legend=alt.Legend(title=None, orient="top")),
                tooltip=["target", "method", alt.Tooltip("accuracy:Q", format=".0%")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width="stretch")
        only_err = st.toggle("Show only sentences Lesk got wrong", value=False, key="only_err")
        shown = ev[~ev["Lesk correct"]] if only_err else ev
        st.dataframe(shown, hide_index=True)

st.caption("WordNet 3.0 · spaCy tagging · extended Lesk")
