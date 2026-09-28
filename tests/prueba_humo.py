import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import PegasusConfig, PegasusForConditionalGeneration

import core


# PEGASUS en miniatura: misma estructura, ~0,1 M parametros en vez de 571 M.
CONFIG = dict(
    vocab_size=200, d_model=32, encoder_layers=2, decoder_layers=2,
    encoder_attention_heads=4, decoder_attention_heads=4,
    encoder_ffn_dim=64, decoder_ffn_dim=64,
    max_position_embeddings=64, pad_token_id=0, eos_token_id=1,
    decoder_start_token_id=0,
)

L_DOC, L_RES = 20, 12


def construir(implementacion):
    torch.manual_seed(0)
    mod = PegasusForConditionalGeneration(
        PegasusConfig(**CONFIG, attn_implementation=implementacion)
    )
    mod.eval()
    return mod


def pasada(mod):
    """Replica lo que hace core.resumir(): hooks + forward, sin tokenizador."""
    ids = torch.randint(2, 200, (1, L_DOC))
    dec = torch.randint(2, 200, (1, L_RES))

    capturas, handles = {}, []

    def crear_hook(i):
        def hook(modulo, args, kwargs, salida):
            h = kwargs.get("hidden_states", args[0] if args else None)
            kv = kwargs.get("key_value_states")
            if kv is None and len(args) > 1:
                kv = args[1]
            capturas[i] = {"q_in": h.detach(), "kv_in": kv.detach()}
        return hook

    for i, capa in enumerate(mod.model.decoder.layers):
        handles.append(
            capa.encoder_attn.register_forward_hook(crear_hook(i), with_kwargs=True)
        )

    with torch.inference_mode():
        salida = mod(input_ids=ids, decoder_input_ids=dec, output_attentions=True)

    for h in handles:
        h.remove()

    return {"salida": salida, "capturas": capturas}


# ---------------------------------------------------------------- pruebas

def prueba_carga_pide_eager():
    """core.cargar() debe pasar attn_implementation='eager' explicitamente."""
    fuente = inspect.getsource(core.cargar)
    assert 'attn_implementation="eager"' in fuente or \
           "attn_implementation='eager'" in fuente, (
        "core.cargar() ya no pide attn_implementation='eager'. Sin eso, "
        "transformers 5.x usa SDPA, devuelve cross_attentions vacio y la app "
        "revienta al calcular Q, K y V."
    )
    print("  OK  core.cargar() pide eager")


def prueba_eager_reproduce_la_atencion():
    """Con eager, el calculo manual debe coincidir con el del modelo."""
    mod = construir("eager")
    res = pasada(mod)
    qkv = core.calcular_qkv(mod, res, capa=0)

    nh, dk = CONFIG["decoder_attention_heads"], CONFIG["d_model"] // CONFIG["decoder_attention_heads"]
    assert tuple(qkv["Q"].shape) == (1, nh, L_RES, dk), f"Q mal formado: {qkv['Q'].shape}"
    assert tuple(qkv["K"].shape) == (1, nh, L_DOC, dk), f"K mal formado: {qkv['K'].shape}"
    assert tuple(qkv["V"].shape) == (1, nh, L_DOC, dk), f"V mal formado: {qkv['V'].shape}"

    filas = qkv["pesos"].sum(dim=-1)
    assert torch.allclose(filas, torch.ones_like(filas), atol=1e-5), \
        "Las filas de los pesos de atencion no suman 1"

    assert qkv["error"] < 1e-5, (
        f"El calculo manual de Q,K,V ya no coincide con el modelo "
        f"(error {qkv['error']:.2e}). Revisa los hooks: PEGASUS usa "
        f"pre-layer-norm y el estado que entra a la atencion cruzada no es "
        f"el que entra a la capa."
    )
    print(f"  OK  eager: error {qkv['error']:.2e}, Q{tuple(qkv['Q'].shape)} "
          f"K{tuple(qkv['K'].shape)} V{tuple(qkv['V'].shape)}")


def prueba_sdpa_falla_con_mensaje_util():
    """Con SDPA el fallo debe ser explicito, no un IndexError opaco."""
    try:
        mod = construir("sdpa")
    except ValueError as e:
        # transformers 4.x no implementa SDPA para PEGASUS: siempre usa eager.
        print(f"  --  esta version no soporta SDPA para PEGASUS ({type(e).__name__})")
        return

    res = pasada(mod)
    if res["salida"].cross_attentions:
        # Alguna version futura podria devolver atenciones tambien con SDPA.
        print("  --  esta version devuelve cross_attentions incluso con SDPA")
        return

    try:
        core.calcular_qkv(mod, res, capa=0)
    except RuntimeError as e:
        assert "eager" in str(e), f"El mensaje de error no menciona eager: {e}"
        print("  OK  sdpa: detectado y con mensaje util")
        return
    raise AssertionError(
        "Con SDPA, calcular_qkv deberia lanzar un RuntimeError explicando que "
        "hay que cargar el modelo con attn_implementation='eager'."
    )


if __name__ == "__main__":
    import transformers
    print(f"=== transformers {transformers.__version__} / torch {torch.__version__} "
          f"/ python {sys.version.split()[0]} ===")
    for prueba in (prueba_carga_pide_eager,
                   prueba_eager_reproduce_la_atencion,
                   prueba_sdpa_falla_con_mensaje_util):
        prueba()
    print("TODAS LAS PRUEBAS PASARON")