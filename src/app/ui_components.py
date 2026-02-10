"""Shared UI components for Streamlit app."""
import streamlit as st


def render_metric(name: str, value: float, format_str: str = "%.2f") -> None:
    """Display a single metric."""
    st.metric(name, format_str % value)


def render_propensity(score: float) -> None:
    """Display propensity score with a progress bar."""
    st.subheader("Propensity score")
    st.progress(min(1.0, max(0.0, score)))
    st.caption(f"P(accept offer) = **{score:.3f}**")


def render_offer_cards(ranked: list[dict]) -> None:
    """Display ranked offers as cards (list of {rank, offer_id, label, description, score})."""
    for i, item in enumerate(ranked, start=1):
        with st.container():
            st.markdown(f"**#{i} — {item.get('label', item.get('offer_id', 'Offer'))}**")
            if item.get("description"):
                st.caption(item["description"])
            st.caption(f"Score: {item.get('score', 0):.3f}")
            st.divider()
