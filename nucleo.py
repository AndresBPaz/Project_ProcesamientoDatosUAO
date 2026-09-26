import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODELO = "google/pegasus-cnn_dailymail"


def cargar(nombre=MODELO):
    tok = AutoTokenizer.from_pretrained(nombre)
    mod = AutoModelForSeq2SeqLM.from_pretrained(nombre)
    mod.eval()
    return tok, mod


@torch.inference_mode()
def resumir(tok, mod, texto, num_beams=4, length_penalty=0.8, max_length=128):
    """ENTRADA: documento de texto. SALIDA: resumen generado + tensores internos."""
    entradas = tok(texto, return_tensors="pt", truncation=True, max_length=1024)

    t0 = time.time()
    generados = mod.generate(**entradas, num_beams=num_beams,
                             length_penalty=length_penalty,
                             max_length=max_length, early_stopping=True)
    dt = time.time() - t0
    resumen = tok.decode(generados[0], skip_special_tokens=True).replace("<n>", " ").strip()

    # Hooks: capturan los tensores EXACTOS que recibe cada modulo de atencion
    # cruzada. Es la unica forma fiable de reproducir Q, K y V, porque PEGASUS
    # usa pre-layer-norm y el estado que entra a la atencion cruzada no es el
    # mismo que entra a la capa.
    capturas = {}
    handles = []

    def crear_hook(i):
        def hook(modulo, args, kwargs, salida):
            h = kwargs.get("hidden_states", args[0] if args else None)
            kv = kwargs.get("key_value_states")
            if kv is None and len(args) > 1:
                kv = args[1]
            capturas[i] = {"q_in": h.detach(), "kv_in": kv.detach()}
        return hook

    for i, capa in enumerate(mod.model.decoder.layers):
        handles.append(capa.encoder_attn.register_forward_hook(crear_hook(i), with_kwargs=True))

    # Segunda pasada con el resumen ya generado: durante beam search los pesos
    # vienen entremezclados entre haces, por eso se recalculan aqui.
    salida = mod(input_ids=entradas["input_ids"],
                 attention_mask=entradas["attention_mask"],
                 decoder_input_ids=generados[:, :-1],
                 output_attentions=True)

    for h in handles:
        h.remove()

    return {"resumen": resumen, "entradas": entradas, "generados": generados,
            "salida": salida, "capturas": capturas, "tiempo": dt,
            "tokens_entrada": int(entradas["input_ids"].shape[1]),
            "tokens_salida": int(generados.shape[1])}


@torch.inference_mode()
def calcular_qkv(mod, res, capa):
    """Reproduce a mano softmax(QK^T/sqrt(d_k))V de la ATENCION CRUZADA.

    Q sale del resumen en construccion; K y V salen del documento de entrada.
    El resultado se verifica contra los pesos que reporta el propio modelo.
    """
    atn = mod.model.decoder.layers[capa].encoder_attn
    cap = res["capturas"][capa]

    nh = getattr(atn, "num_heads", mod.config.decoder_attention_heads)
    dk = getattr(atn, "head_dim", mod.config.d_model // nh)
    escala = float(getattr(atn, "scaling", dk ** -0.5))

    def partir(x):
        b, l, _ = x.shape
        return x.view(b, l, nh, dk).transpose(1, 2)   # (1, cabezas, L, d_k)

    Q = partir(atn.q_proj(cap["q_in"]))               # del RESUMEN
    K = partir(atn.k_proj(cap["kv_in"]))              # del DOCUMENTO
    V = partir(atn.v_proj(cap["kv_in"]))              # del DOCUMENTO

    puntajes = (Q @ K.transpose(-1, -2)) * escala     # QK^T / sqrt(d_k)
    pesos = torch.softmax(puntajes, dim=-1)           # distribucion por fila
    contexto = pesos @ V                              # promedio ponderado de V

    error = (pesos - res["salida"].cross_attentions[capa]).abs().max().item()

    return {"Q": Q, "K": K, "V": V, "pesos": pesos, "contexto": contexto,
            "error": error, "d_k": dk, "n_cabezas": nh, "escala": escala,
            "forma_wq": tuple(atn.q_proj.weight.shape)}


def tokens_legibles(tok, ids):
    return [t.replace("▁", " ").strip() or t for t in tok.convert_ids_to_tokens(ids)]
