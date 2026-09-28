import resource
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# --- diagnostico temporal: cuanta memoria se usa y hasta donde llega el arranque
# En Linux, ru_maxrss viene en kilobytes.
def _diag(etapa):
    mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[diag] {etapa} | RSS maximo {mb:.0f} MB", flush=True)
    sys.stdout.flush()


_diag("1. streamlit importado")

import core as N

_diag("2. core y torch importados")

st.set_page_config(page_title="PEGASUS - Resumen abstractivo", layout="wide")


@st.cache_resource(show_spinner="Cargando PEGASUS...")
def _cargar():
    _diag("3. antes de descargar/cargar los pesos")
    r = N.cargar()
    _diag("4. modelo cargado en memoria")
    return r


tokenizer, modelo = _cargar()
_diag("5. app lista")
cfg = modelo.config

st.title("PEGASUS: resumen abstractivo de documentos")
st.caption("Zhang, Zhao, Saleh y Liu (2020), ICML 2020 - arXiv:1912.08777  |  "
           "checkpoint `google/pegasus-cnn_dailymail`")

with st.sidebar:
    st.subheader("Arquitectura")
    st.table(pd.DataFrame({
        "Valor": [cfg.encoder_layers, cfg.decoder_layers, cfg.d_model,
                  cfg.decoder_attention_heads, cfg.d_model // cfg.decoder_attention_heads,
                  cfg.decoder_ffn_dim, f"{cfg.vocab_size:,}", cfg.max_position_embeddings,
                  f"{sum(p.numel() for p in modelo.parameters())/1e6:.0f} M"]},
        index=["Capas encoder", "Capas decoder", "d_model", "Cabezas", "d_k",
               "Feed-forward", "Vocabulario", "Entrada max.", "Parametros"]))

    st.subheader("Decodificacion")
    num_beams = st.slider("num_beams", 1, 8, 4)
    length_penalty = st.slider("length_penalty", 0.2, 2.0, 0.8, 0.1)
    max_length = st.slider("max_length", 32, 200, 128, 8)

EJEMPLO = ("The World Health Organization announced on Monday that global life expectancy has "
           "increased by more than six years since the year 2000, reaching an average of 73.3 years. "
           "The agency attributed the improvement to better access to vaccines, cleaner drinking "
           "water and reductions in child mortality across low-income countries. However, the report "
           "also warned that progress has slowed considerably since 2019, when the COVID-19 pandemic "
           "reversed several health gains and overwhelmed hospital systems worldwide. Researchers "
           "noted that the gap between the richest and poorest nations remains wide, with a difference "
           "of nearly thirty years in life expectancy between the top and bottom ranked countries. "
           "Non-communicable diseases such as diabetes and heart conditions are now the leading cause "
           "of death in most regions, overtaking infectious diseases for the first time in history.")

# --------------------------------------------------------------- 1. Entrada
st.subheader("1. Entrada")
ca, cb = st.columns([3, 1])
with ca:
    texto = st.text_area("Documento (ingles)", value=EJEMPLO, height=180,
                         label_visibility="collapsed")
with cb:
    archivo = st.file_uploader("Cargar .txt", type=["txt"])
    if archivo is not None:
        texto = archivo.read().decode("utf-8", errors="ignore")
    st.metric("Palabras", len(texto.split()))
    ejecutar = st.button("Generar resumen", type="primary", use_container_width=True)

if ejecutar or "res" not in st.session_state:
    if texto.strip():
        with st.spinner("Inferencia..."):
            st.session_state.res = N.resumir(tokenizer, modelo, texto,
                                             num_beams, length_penalty, max_length)

res = st.session_state.get("res")

if res:
    # ----------------------------------------------------------- 2. Salida
    st.subheader("2. Salida")
    st.success(res["resumen"])
    m = st.columns(4)
    m[0].metric("Tokens entrada", res["tokens_entrada"])
    m[1].metric("Tokens salida", res["tokens_salida"])
    m[2].metric("Compresion", f"{res['tokens_entrada']/max(res['tokens_salida'],1):.1f}x")
    m[3].metric("Tiempo", f"{res['tiempo']:.1f} s")

    # ------------------------------------------------------------ 3. Q K V
    st.subheader("3. Q, K y V en la atencion cruzada")
    s1, s2 = st.columns(2)
    capa = s1.selectbox("Capa del decoder", list(range(cfg.decoder_layers)),
                        index=cfg.decoder_layers - 1)
    cabeza = s2.selectbox("Cabeza", list(range(cfg.decoder_attention_heads)), index=0)

    q = N.calcular_qkv(modelo, res, capa)

    izq, der = st.columns([1.15, 1])
    with izq:
        st.table(pd.DataFrame({
            "Sale de": ["resumen en construccion", "documento de entrada",
                        "documento de entrada", "softmax(QKt/sqrt(d_k))", "pesos x V"],
            "Forma": [str(tuple(q["Q"].shape)), str(tuple(q["K"].shape)),
                      str(tuple(q["V"].shape)), str(tuple(q["pesos"].shape)),
                      str(tuple(q["contexto"].shape))]},
            index=["Q (query)", "K (key)", "V (value)", "Pesos", "Contexto"]))
    with der:
        st.latex(r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left("
                 r"\frac{QK^{\top}}{\sqrt{d_k}}\right)V")
        st.metric("Error |calculo manual - modelo|", f"{q['error']:.1e}")
        st.caption(f"d_k = {q['d_k']}  |  factor de escala 1/sqrt(d_k) = {q['escala']:.4f}")
        st.caption(f"Q, K y V se obtienen con q_proj, k_proj y v_proj "
                   f"({q['forma_wq'][0]}x{q['forma_wq'][1]} cada una), repartidas en "
                   f"{q['n_cabezas']} cabezas de {q['d_k']} dimensiones. El error es la "
                   f"diferencia entre el calculo manual y los pesos que reporta el modelo.")

    # -------------------------------------------------------- mapa de calor
    te = N.tokens_legibles(tokenizer, res["entradas"]["input_ids"][0].tolist())
    ts = N.tokens_legibles(tokenizer, res["generados"][0, :-1].tolist())
    ne, ns = min(len(te), 42), min(len(ts), 26)

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(q["pesos"][0, cabeza][:ns, :ne].float().numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(ne)); ax.set_xticklabels(te[:ne], rotation=90, fontsize=7)
    ax.set_yticks(range(ns)); ax.set_yticklabels(ts[:ns], fontsize=7)
    ax.set_xlabel("DOCUMENTO (K y V)")
    ax.set_ylabel("RESUMEN (Q)")
    ax.set_title(f"Atencion cruzada - capa {capa}, cabeza {cabeza}")
    fig.colorbar(im, ax=ax, label="peso")
    fig.tight_layout()
    st.pyplot(fig)