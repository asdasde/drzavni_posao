from __future__ import annotations

import os
from pathlib import Path

try:
    import streamlit as st
except ImportError as error:  # pragma: no cover - exercised only without UI extra
    raise RuntimeError(
        "The web interface needs Streamlit. Install the 'search' optional dependencies."
    ) from error

from drzavni_posao.search.retrieval.engine import SearchEngine


SEGMENTS_PATH = os.environ.get("DP_SEGMENTS_PATH", "data/segments.jsonl")
BM25_PATH = os.environ.get("DP_BM25_PATH", "artifacts/bm25_index.json")
VECTOR_PATH = os.environ.get("DP_VECTOR_PATH", "artifacts/vector_index")


def _vector_metadata_exists(path: str) -> bool:
    candidate = Path(path)
    return (
        candidate if candidate.suffix == ".json" else candidate.with_suffix(".json")
    ).exists()


def _default_method_index(methods: list[str]) -> int:
    return methods.index("BM25")


@st.cache_resource(show_spinner="Loading search indexes…")
def load_engine(use_vectors: bool) -> SearchEngine:
    return SearchEngine.from_paths(
        segments_path=SEGMENTS_PATH,
        bm25_path=BM25_PATH,
        vector_path=VECTOR_PATH if use_vectors else None,
        local_models_only=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Državni posao — pretraga scena",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Pretraga scena — Državni posao")
    st.caption(
        "Opišite scenu na srpskom ili engleskom. Rezultati prikazuju relativnu "
        "relevantnost i originalni transkript kao dokaz."
    )

    vector_available = _vector_metadata_exists(VECTOR_PATH)
    methods = ["BM25"]
    if vector_available:
        methods.extend(["Dense", "Hybrid"])

    with st.sidebar:
        st.header("Podešavanja")
        method_label = st.selectbox(
            "Metod pretrage",
            methods,
            index=_default_method_index(methods),
            help="Hybrid kombinuje ključne reči, semantiku i imena likova.",
        )
        top_k = st.slider("Broj epizoda", min_value=1, max_value=20, value=5)
        candidate_k = st.slider(
            "Kandidati po indeksu", min_value=10, max_value=100, value=50, step=10
        )
        group_by_episode = st.checkbox("Jedan rezultat po epizodi", value=True)
        if not vector_available:
            st.info(
                "Vektorski indeks nije pronađen. BM25 radi samostalno; izgradite "
                "vektore da biste uključili semantičku i hibridnu pretragu."
            )

    with st.form("scene-search"):
        query = st.text_input(
            "Opis scene",
            placeholder="epizoda u kojoj se Boškić i Čvarkov svađaju oko izbora",
        )
        submitted = st.form_submit_button("Pretraži", type="primary")

    if not submitted:
        st.markdown(
            "Primeri: `Čvarkov priča o izborima`, `the scene about a broken "
            "computer`, `Торбица се жали на плату`."
        )
        return
    if not query.strip():
        st.warning("Unesite opis scene.")
        return

    method = method_label.casefold()
    try:
        engine = load_engine(use_vectors=method != "bm25")
        with st.spinner("Pretražujem transkripte…"):
            results = engine.search(
                query,
                method=method,  # type: ignore[arg-type]
                top_k=top_k,
                candidate_k=candidate_k,
                group_by_episode=group_by_episode,
            )
    except Exception as error:  # UI boundary: turn backend failures into guidance
        st.error(f"Pretraga nije dostupna: {error}")
        st.caption(
            "Proverite putanje do artefakata i da li je model sačuvan lokalno. "
            "Aplikacija namerno ne preuzima modele bez izričite dozvole."
        )
        return

    if not results:
        st.info("Nema rezultata za ovaj upit.")
        return

    st.subheader(f"Najverovatniji rezultati ({len(results)})")
    if not any(result.youtube_url_with_timestamp for result in results):
        st.warning(
            "Rezultati imaju tačne vremenske oznake, ali izvorni YouTube linkovi "
            "nisu upisani u podatke, pa video ne može da se prikaže na stranici."
        )
    for result in results:
        segment = result.segment
        with st.container(border=True):
            title_column, score_column = st.columns([4, 1])
            with title_column:
                st.markdown(
                    f"### {result.rank}. {segment.episode_id} — {segment.episode_title}"
                )
                st.write(
                    f"**Vreme u videu:** {segment.video_start_time}–{segment.video_end_time}"
                )
                st.write(
                    f"**Likovi:** {', '.join(segment.speakers) if segment.speakers else 'nepoznato'}"
                )
            with score_column:
                st.metric("Relativni skor", f"{result.score:.3f}")
            st.write(f"**Zašto:** {result.reason}")
            if result.youtube_url_with_timestamp and segment.youtube_url:
                with st.expander(
                    f"▶ Pogledaj scenu od {segment.video_start_time}",
                    expanded=False,
                ):
                    st.video(
                        segment.youtube_url,
                        start_time=max(0, int(round(segment.video_start_seconds))),
                    )
                    st.link_button(
                        "Otvori na YouTube-u",
                        result.youtube_url_with_timestamp,
                    )
            st.text(segment.text_original)
            with st.expander("Komponente skora"):
                st.json(result.score_components)


if __name__ == "__main__":
    main()
