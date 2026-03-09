from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.inference import score_customers

# ── Design tokens ─────────────────────────────────────────────────────────────
RISK_COLORS = {
    "alto":  "#f43f5e",
    "medio": "#f59e0b",
    "baixo": "#10b981",
}

RISK_BG = {
    "alto":  "rgba(244,63,94,0.12)",
    "medio": "rgba(245,158,11,0.12)",
    "baixo": "rgba(16,185,129,0.12)",
}

RISK_LABEL = {
    "alto":  "Alto",
    "medio": "Médio",
    "baixo": "Baixo",
}

# ── Theme CSS ──────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Epilogue:wght@500;700;900&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bg0:       #0d0f12;
    --bg1:       #13161b;
    --bg2:       #1a1e26;
    --bg3:       #222733;
    --border:    rgba(255,255,255,0.07);
    --border-md: rgba(255,255,255,0.13);

    --jade:      #00c896;
    --jade-dim:  rgba(0,200,150,0.13);
    --jade-glow: rgba(0,200,150,0.28);

    --text1: #f0f2f5;
    --text2: #9ba3b2;
    --text3: #5e6674;

    --danger:  #f43f5e;
    --warning: #f59e0b;
    --safe:    #10b981;

    --r: 12px;
    --r-pill: 999px;
}

html, body, [class*="css"] {
    font-family: "IBM Plex Sans", sans-serif;
    color: var(--text1);
    -webkit-font-smoothing: antialiased;
}

.stApp { background: var(--bg0); }

.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px;
    animation: page-in 340ms ease-out;
}

@keyframes page-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg1) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text1) !important; }
[data-testid="stSidebar"] label {
    color: var(--text2) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

/* ── Buttons ── */
div.stButton > button {
    background: var(--bg3) !important;
    color: var(--text1) !important;
    border: 1px solid var(--border-md) !important;
    border-radius: 8px !important;
    font-family: "IBM Plex Sans", sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    transition: all 0.15s !important;
}
div.stButton > button:hover {
    border-color: var(--jade) !important;
    color: var(--jade) !important;
    background: var(--jade-dim) !important;
}
div.stButton > button[kind="primary"] {
    background: var(--jade) !important;
    color: #021a0f !important;
    border: none !important;
    box-shadow: 0 0 22px var(--jade-glow) !important;
    font-weight: 700 !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #00dea8 !important;
    box-shadow: 0 0 32px var(--jade-glow) !important;
}

/* ── Inputs ── */
div.stTextInput input,
div.stNumberInput input {
    background: var(--bg2) !important;
    border: 1px solid var(--border-md) !important;
    color: var(--text1) !important;
    border-radius: 8px !important;
}
div.stTextInput input:focus,
div.stNumberInput input:focus {
    border-color: var(--jade) !important;
    box-shadow: 0 0 0 2px var(--jade-dim) !important;
}
div.stSelectbox > div > div,
div.stMultiSelect > div > div {
    background: var(--bg2) !important;
    border: 1px solid var(--border-md) !important;
    color: var(--text1) !important;
    border-radius: 8px !important;
}

/* ── Labels ── */
label, .stRadio label, .stCheckbox label {
    color: var(--text2) !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    overflow: hidden;
}

/* ── Alerts ── */
div.stAlert {
    background: var(--bg2) !important;
    border: 1px solid var(--border-md) !important;
    border-radius: var(--r) !important;
    color: var(--text1) !important;
}

/* ── File uploader ── */
div.stFileUploader > div {
    background: var(--bg2) !important;
    border: 1.5px dashed var(--border-md) !important;
    border-radius: var(--r) !important;
}

/* ── Caption ── */
.stCaption, small { color: var(--text3) !important; font-size: 0.74rem !important; }
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg1); }
::-webkit-scrollbar-thumb { background: var(--bg3); border-radius: 3px; }

/* ── Custom components ── */
.hero {
    position: relative;
    background: var(--bg1);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 700px 280px at 100% 50%, rgba(0,200,150,0.07), transparent 65%),
        radial-gradient(ellipse 320px 180px at 0% 0%,   rgba(0,200,150,0.04), transparent 60%);
    pointer-events: none;
}

.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--jade);
    background: var(--jade-dim);
    border: 1px solid rgba(0,200,150,0.2);
    border-radius: var(--r-pill);
    padding: 0.2rem 0.65rem;
    margin-bottom: 0.7rem;
}
.hero-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--jade); box-shadow: 0 0 6px var(--jade);
    animation: blink 2.2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }

.hero h1 {
    margin: 0 0 0.4rem; font-family: "Epilogue", sans-serif;
    font-size: 2.1rem; font-weight: 900;
    letter-spacing: -0.045em; line-height: 1; color: var(--text1);
}
.hero h1 em { font-style: normal; color: var(--jade); }
.hero p { margin: 0; color: var(--text2); font-size: 0.88rem; line-height: 1.65; max-width: 560px; }

.kpi {
    background: var(--bg1); border: 1px solid var(--border);
    border-radius: var(--r); padding: 1.05rem 1.15rem;
    min-height: 112px; position: relative; overflow: hidden;
    transition: border-color 0.18s, transform 0.15s;
}
.kpi:hover { border-color: var(--border-md); transform: translateY(-1px); }
.kpi-bar {
    position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px; border-radius: 3px 0 0 3px;
}
.kpi-label {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.66rem; font-weight: 600;
    letter-spacing: 0.11em; text-transform: uppercase;
    color: var(--text3); margin-bottom: 0.45rem;
}
.kpi-value {
    font-family: "Epilogue", sans-serif;
    font-size: 2.05rem; font-weight: 900;
    letter-spacing: -0.04em; line-height: 1;
    color: var(--text1); margin-bottom: 0.3rem;
}
.kpi-hint { font-size: 0.77rem; color: var(--text2); font-weight: 500; }

.sec {
    display: flex; align-items: center; gap: 0.55rem;
    margin: 1.1rem 0 0.65rem;
}
.sec-title {
    font-family: "Epilogue", sans-serif;
    font-size: 0.9rem; font-weight: 700;
    color: var(--text1); white-space: nowrap;
}
.sec-rule { flex: 1; height: 1px; background: var(--border); }

.shell-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--r); padding: 0.9rem 1rem; margin: 0.4rem 0;
    transition: border-color 0.18s;
}
.shell-card:hover { border-color: var(--border-md); }

.p-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--r); padding: 0.8rem 0.95rem;
    margin-bottom: 0.45rem;
    transition: border-color 0.18s, background 0.15s;
}
.p-card:hover { border-color: var(--border-md); background: var(--bg3); }
.p-card-top {
    display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 0.38rem;
}
.p-id {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.8rem; font-weight: 600; color: var(--text1);
}
.chip {
    display: inline-flex; align-items: center; gap: 0.25rem;
    border-radius: var(--r-pill); padding: 0.17rem 0.58rem;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem; font-weight: 600;
}
.p-action { font-size: 0.8rem; color: var(--text2); line-height: 1.45; }

.sb-brand {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--r); padding: 0.88rem 1rem; margin-bottom: 0.8rem;
}
.sb-title {
    font-family: "Epilogue", sans-serif; font-size: 1rem;
    font-weight: 900; letter-spacing: -0.03em; color: var(--text1);
    margin: 0 0 0.15rem;
}
.sb-title em { font-style: normal; color: var(--jade); }
.sb-sub { font-size: 0.76rem; color: var(--text3); margin: 0; font-weight: 500; }

.status-live {
    display: flex; align-items: center; justify-content: center; gap: 0.4rem;
    background: var(--jade-dim); border: 1px solid rgba(0,200,150,0.2);
    color: var(--jade); border-radius: var(--r-pill);
    padding: 0.3rem 0.8rem; font-family: "IBM Plex Mono", monospace;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em;
    width: 100%; box-sizing: border-box;
}
.status-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--jade); box-shadow: 0 0 5px var(--jade);
}
.status-idle {
    display: flex; align-items: center; justify-content: center;
    color: var(--text3); font-size: 0.76rem; font-weight: 500;
    padding: 0.3rem 0; width: 100%;
}

.step {
    display: flex; align-items: flex-start; gap: 0.8rem;
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--r); padding: 0.85rem 0.95rem; margin-bottom: 0.42rem;
}
.step-n {
    width: 24px; height: 24px; border-radius: 6px; flex-shrink: 0;
    background: var(--jade-dim); border: 1px solid rgba(0,200,150,0.2);
    color: var(--jade); font-family: "Epilogue", sans-serif;
    font-weight: 900; font-size: 0.8rem;
    display: flex; align-items: center; justify-content: center;
}
.step-t { font-weight: 600; font-size: 0.87rem; color: var(--text1); margin-bottom: 0.16rem; }
.step-d { font-size: 0.8rem; color: var(--text2); line-height: 1.5; }

.field-group {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.11em; text-transform: uppercase;
    color: var(--text3); margin-bottom: 0.6rem; margin-top: 0.2rem;
}
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def init_state() -> None:
    st.session_state.setdefault("input_df", None)
    st.session_state.setdefault("scores_df", None)
    st.session_state.setdefault("last_scored", None)


def risk_color(level: str) -> str:
    return RISK_COLORS.get(level, RISK_COLORS["baixo"])


def risk_bg(level: str) -> str:
    return RISK_BG.get(level, RISK_BG["baixo"])


def try_score(df: pd.DataFrame, model_path: str) -> pd.DataFrame | None:
    try:
        return score_customers(df=df, model_path=model_path, output_path=None)
    except FileNotFoundError:
        st.error("Modelo não encontrado. Treine com `python src/main.py train --train-path data/raw/churn_train.csv`.")
    except Exception as exc:
        st.error(f"Falha ao calcular score: {exc}")
    return None


# ── Components ─────────────────────────────────────────────────────────────────

def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow"><div class="hero-dot"></div>Retenção · Operacional</div>
            <h1>Churn<em>Copilot</em></h1>
            <p>Priorize clientes em risco, oriente ações comerciais e monitore a saúde da base — com precisão, em tempo real.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="sec"><span class="sec-title">{icon}&ensp;{title}</span><div class="sec-rule"></div></div>',
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str, hint: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-bar" style="background:{accent};"></div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(scores_df: pd.DataFrame) -> None:
    total     = len(scores_df)
    high      = int((scores_df["risk_level"] == "alto").sum())
    medium    = int((scores_df["risk_level"] == "medio").sum())
    low       = int((scores_df["risk_level"] == "baixo").sum())
    avg_score = float(scores_df["churn_risk_score"].mean())

    c1, c2, c3, c4 = st.columns(4)
    with c1: render_kpi("Base avaliada", f"{total:,}".replace(",", "."), "clientes carregados", "#00c896")
    with c2: render_kpi("Risco alto",    str(high),   "ação imediata",   "#f43f5e")
    with c3: render_kpi("Risco médio",   str(medium), "monitorar",       "#f59e0b")
    with c4: render_kpi("Score médio",   f"{avg_score:.1%}", f"baixo risco: {low}", "#10b981")


def render_risk_chart(scores_df: pd.DataFrame) -> None:
    ordered = ["alto", "medio", "baixo"]
    dist = (
        scores_df["risk_level"]
        .value_counts()
        .reindex(ordered, fill_value=0)
        .rename_axis("risk_level")
        .reset_index(name="customers")
    )
    dist["label"] = dist["risk_level"].map({"alto": "Alto", "medio": "Médio", "baixo": "Baixo"})

    fig = px.bar(dist, x="label", y="customers", color="risk_level",
                 text_auto=True, color_discrete_map=RISK_COLORS)
    fig.update_traces(marker_line_width=0,
                      textfont=dict(family="IBM Plex Mono", size=13, color="#f0f2f5"))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=16, r=6, l=6, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", color="#9ba3b2"),
        xaxis=dict(title=None, tickfont=dict(size=12, color="#9ba3b2"), gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="clientes", tickfont=dict(size=11, color="#5e6674"), gridcolor="rgba(255,255,255,0.05)"),
        bargap=0.4,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_priority_cards(scores_df: pd.DataFrame) -> None:
    section("⚡", "Prioridades agora")
    for _, row in scores_df.head(5).iterrows():
        level = str(row["risk_level"])
        color = risk_color(level)
        bg    = risk_bg(level)
        score = float(row["churn_risk_score"])
        label = RISK_LABEL.get(level, level)
        st.markdown(
            f"""
            <div class="p-card">
                <div class="p-card-top">
                    <span class="p-id">#{row['customer_id']}</span>
                    <span class="chip" style="background:{bg};color:{color};border:1px solid {color}33;">
                        {label} · {score:.1%}
                    </span>
                </div>
                <div class="p-action">{row['retention_action']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_operations(scores_df: pd.DataFrame) -> None:
    section("⚙️", "Fila de Retenção")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 1.2, 1.8])
        with c1:
            levels = st.multiselect("Níveis de risco", options=["alto", "medio", "baixo"], default=["alto", "medio"])
        with c2:
            min_score = st.slider("Score mínimo", 0.0, 1.0, 0.4, 0.01)
        with c3:
            customer_search = st.text_input("Buscar customer_id", value="", placeholder="Filtrar por ID...")

    filtered = scores_df[
        (scores_df["risk_level"].isin(levels)) &
        (scores_df["churn_risk_score"] >= min_score)
    ].copy()

    if customer_search.strip():
        q = customer_search.strip().lower()
        filtered = filtered[filtered["customer_id"].astype(str).str.lower().str.contains(q, na=False)]

    shown = filtered.copy()
    shown["churn_risk_score"] = (shown["churn_risk_score"] * 100).round(1).astype(str) + "%"
    st.dataframe(shown, use_container_width=True, hide_index=True)

    cols = st.columns([3, 1])
    with cols[0]:
        high_c   = int((filtered["risk_level"] == "alto").sum())
        medium_c = int((filtered["risk_level"] == "medio").sum())
        low_c    = int((filtered["risk_level"] == "baixo").sum())
        st.caption(f"{len(filtered)} clientes  ·  Alto {high_c}  ·  Médio {medium_c}  ·  Baixo {low_c}")
    with cols[1]:
        st.download_button(
            "↓ Exportar CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="fila_retencao.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_simulator(model_path: str) -> None:
    section("🔬", "Simulador Individual")
    with st.form("simulator_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="field-group">Identificação</div>', unsafe_allow_html=True)
            customer_id            = st.text_input("Customer ID", value="cliente-001")
            tenure_months          = st.number_input("Tempo de base (meses)", 0.0, 120.0, 14.0)
            recency_days           = st.number_input("Recência (dias)", 0, 365, 40)
            purchase_frequency_90d = st.number_input("Compras 90d", 0, 300, 3)
            avg_ticket             = st.number_input("Ticket médio", 0.0, 100000.0, 130.0)
        with c2:
            st.markdown('<div class="field-group">Comportamento</div>', unsafe_allow_html=True)
            support_tickets_90d = st.number_input("Tickets suporte 90d", 0, 200, 2)
            payment_delay_days  = st.number_input("Atraso pagamento (dias)", 0, 365, 4)
            failed_payments_90d = st.number_input("Falhas pagamento 90d", 0, 50, 1)
            login_days_30d      = st.number_input("Dias de login 30d", 0, 30, 10)
            engagement_30d      = st.number_input("Engajamento 30d", 0.0, 1.0, 0.33)
        with c3:
            st.markdown('<div class="field-group">Contrato & Plano</div>', unsafe_allow_html=True)
            usage_ratio        = st.number_input("Uso ratio", 0.0, 1.0, 0.45)
            nps_score          = st.number_input("NPS", 0, 10, 6)
            satisfaction_score = st.number_input("Satisfação", 1, 5, 3)
            plan_value         = st.number_input("Valor do plano", 0.0, 20000.0, 99.9)
            plan_type          = st.selectbox("Plano", ["basico", "padrao", "premium"], index=1)
            contract_type      = st.selectbox("Contrato", ["mensal", "anual"], index=0)
            payment_method     = st.selectbox("Pagamento", ["cartao", "boleto", "pix", "debito"], index=0)
            region             = st.selectbox("Região", ["sudeste", "sul", "nordeste", "norte", "centro-oeste"], index=0)

        submitted = st.form_submit_button("Executar simulação", use_container_width=True, type="primary")

    if not submitted:
        return

    payload: dict[str, Any] = {
        "customer_id": customer_id, "tenure_months": tenure_months,
        "recency_days": recency_days, "purchase_frequency_90d": purchase_frequency_90d,
        "avg_ticket": avg_ticket, "support_tickets_90d": support_tickets_90d,
        "payment_delay_days": payment_delay_days, "failed_payments_90d": failed_payments_90d,
        "login_days_30d": login_days_30d, "engagement_30d": engagement_30d,
        "usage_ratio": usage_ratio, "nps_score": nps_score,
        "satisfaction_score": satisfaction_score, "plan_value": plan_value,
        "plan_type": plan_type, "contract_type": contract_type,
        "payment_method": payment_method, "region": region,
    }

    result = try_score(pd.DataFrame([payload]), model_path=model_path)
    if result is None or result.empty:
        return

    score  = float(result.loc[0, "churn_risk_score"])
    level  = str(result.loc[0, "risk_level"])
    action = str(result.loc[0, "retention_action"])
    color  = risk_color(level)
    bg     = risk_bg(level)
    label  = RISK_LABEL.get(level, level)

    g1, g2 = st.columns([1, 1.2])
    with g1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"family": "Epilogue", "size": 48, "color": "#f0f2f5"}},
            title={"text": "Score de churn", "font": {"family": "IBM Plex Sans", "size": 12, "color": "#9ba3b2"}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"family": "IBM Plex Mono", "color": "#5e6674", "size": 10}},
                "bar": {"color": color, "thickness": 0.16},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range": [0, 40],   "color": "rgba(16,185,129,0.1)"},
                    {"range": [40, 70],  "color": "rgba(245,158,11,0.1)"},
                    {"range": [70, 100], "color": "rgba(244,63,94,0.1)"},
                ],
            },
        ))
        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30, r=16, l=16, b=0),
        )
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

    with g2:
        st.markdown(
            f"""
            <div class="shell-card" style="border-left:3px solid {color}; margin-top:0.4rem;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;font-weight:600;
                            letter-spacing:0.11em;text-transform:uppercase;color:#5e6674;margin-bottom:1rem;">
                    Resultado
                </div>
                <div style="margin-bottom:0.85rem;">
                    <div style="font-size:0.68rem;font-weight:600;color:#5e6674;text-transform:uppercase;
                                letter-spacing:0.1em;margin-bottom:0.2rem;">Cliente</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;
                                font-weight:600;color:#f0f2f5;">{customer_id}</div>
                </div>
                <div style="margin-bottom:0.85rem;">
                    <div style="font-size:0.68rem;font-weight:600;color:#5e6674;text-transform:uppercase;
                                letter-spacing:0.1em;margin-bottom:0.28rem;">Nível de risco</div>
                    <span class="chip" style="background:{bg};color:{color};border:1px solid {color}44;">
                        {label} · {score:.1%}
                    </span>
                </div>
                <div>
                    <div style="font-size:0.68rem;font-weight:600;color:#5e6674;text-transform:uppercase;
                                letter-spacing:0.1em;margin-bottom:0.2rem;">Ação recomendada</div>
                    <div style="font-size:0.84rem;color:#9ba3b2;line-height:1.55;">{action}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_onboarding() -> None:
    section("→", "Como começar")
    steps = [
        ("1", "Carregar dados",      "Use <b style='color:#f0f2f5'>Carregar base exemplo</b> no menu lateral para iniciar com dados de demonstração."),
        ("2", "Executar previsão",   "Clique em <b style='color:#f0f2f5'>Executar previsão</b> — o modelo pontua cada cliente e monta a fila de risco."),
        ("3", "Explorar resultados", "Navegue entre <b style='color:#f0f2f5'>Cockpit</b>, <b style='color:#f0f2f5'>Operação</b> e <b style='color:#f0f2f5'>Simulador</b>."),
    ]
    for num, title, desc in steps:
        st.markdown(
            f"""
            <div class="step">
                <div class="step-n">{num}</div>
                <div>
                    <div class="step-t">{title}</div>
                    <div class="step-d">{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_config(model_path: str) -> None:
    section("🗂", "Configuração")
    uploaded_file = st.file_uploader("Enviar CSV de clientes", type=["csv"], key="config_uploader")
    if uploaded_file is not None:
        try:
            st.session_state["input_df"] = pd.read_csv(uploaded_file)
            st.success("CSV carregado.")
        except Exception as exc:
            st.error(f"Falha ao ler o CSV: {exc}")

    input_df = st.session_state.get("input_df")
    if isinstance(input_df, pd.DataFrame):
        st.markdown(
            f'<div class="shell-card" style="margin-bottom:0.5rem;">'
            f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.8rem;color:#9ba3b2;">'
            f'{len(input_df)} clientes carregados</span></div>',
            unsafe_allow_html=True,
        )
        st.dataframe(input_df.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("Nenhuma base carregada. Use o upload acima ou carregue a base de exemplo.")

    st.caption(f"Modelo: {model_path}")


def load_sample_data() -> None:
    p = Path("data/raw/churn_score.csv")
    if not p.exists():
        st.warning("Base de exemplo não encontrada. Execute `python -m src.data.generate_sample`.")
        return
    st.session_state["input_df"] = pd.read_csv(p)
    st.success("Base carregada.")


def run_scoring(model_path: str) -> None:
    input_df = st.session_state.get("input_df")
    if not isinstance(input_df, pd.DataFrame) or input_df.empty:
        st.warning("Carregue uma base antes de executar a previsão.")
        return
    scored = try_score(input_df, model_path=model_path)
    if scored is not None:
        st.session_state["scores_df"]   = scored
        st.session_state["last_scored"] = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.success("Previsão concluída.")


# ── App shell ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Churn Copilot",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    init_state()

    with st.sidebar:
        st.markdown(
            """
            <div class="sb-brand">
                <div class="sb-title">◈ Churn<em>Copilot</em></div>
                <div class="sb-sub">Previsão · Retenção · Operação</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navegação",
            options=["Cockpit", "Operação", "Simulador", "Configuração"],
            index=0,
            label_visibility="collapsed",
        )

        model_path = st.text_input(
            "Modelo", value="models/churn_model.joblib",
            label_visibility="collapsed",
            placeholder="caminho/para/modelo.joblib",
        )

        st.divider()

        if st.button("Carregar base exemplo", use_container_width=True):
            load_sample_data()
        if st.button("Executar previsão", type="primary", use_container_width=True):
            run_scoring(model_path=model_path)

        st.divider()

        scores_df = st.session_state.get("scores_df")
        if isinstance(scores_df, pd.DataFrame) and not scores_df.empty:
            st.markdown(
                f"""
                <div class="status-live">
                    <div class="status-dot"></div>
                    {len(scores_df)} clientes &nbsp;·&nbsp; {st.session_state.get("last_scored", "")}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="status-idle">Sem previsão ativa</div>', unsafe_allow_html=True)

    render_hero()
    scores_df = st.session_state.get("scores_df")

    if page == "Cockpit":
        if isinstance(scores_df, pd.DataFrame) and not scores_df.empty:
            render_overview(scores_df)
            c1, c2 = st.columns([1.35, 1])
            with c1:
                section("📊", "Distribuição de risco")
                render_risk_chart(scores_df)
            with c2:
                render_priority_cards(scores_df)
        else:
            render_onboarding()

    elif page == "Operação":
        if isinstance(scores_df, pd.DataFrame) and not scores_df.empty:
            render_operations(scores_df)
        else:
            st.warning("Execute uma previsão para abrir a fila de operação.")

    elif page == "Simulador":
        render_simulator(model_path=model_path)

    elif page == "Configuração":
        render_config(model_path=model_path)


if __name__ == "__main__":
    main()