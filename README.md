# Despliegue e interpretación de PEGASUS para resumen abstractivo

Proyecto final — Procesamiento de Datos Secuenciales
Maestría en Inteligencia artificial y ciencias de datos, Universidad Autónoma de Occidente

Integrantes: 

Andres David Bolaños, 
Dilmer Gutierrez
Eliana Ordoñez
Eliana Vanessa Cardona Burbano


## 1. Resumen

Se implementa el proceso de inferencia de **PEGASUS**, un Transformer encoder-decoder para resumen abstractivo, y se despliega en una interfaz interactiva que recibe un documento y devuelve su resumen.

Además de la inferencia, la implementación extrae del modelo las matrices `q_proj`, `k_proj` y `v_proj`, calcula Q, K y V a mano y reconstruye la fórmula de atención. El resultado se compara contra los pesos que reporta el propio modelo: **la diferencia es 0.0 en las 16 capas del decoder**, lo que confirma que la interpretación corresponde al cálculo real.

Sobre un documento de 148 tokens el sistema generó un resumen de 81 tokens (compresión 1,8x) en 4,5 segundos sobre CPU.

---

## 2. Introducción

**Artículo base:** J. Zhang, Y. Zhao, M. Saleh y P. J. Liu, *PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization*, ICML 2020.

- Artículo: https://arxiv.org/abs/1912.08777
- Repositorio original: https://github.com/google-research/pegasus
- Checkpoint: `google/pegasus-cnn_dailymail`

**Problema.** El resumen extractivo copia oraciones del documento; el abstractivo genera texto nuevo, como haría una persona. El segundo es más difícil y es el que aborda PEGASUS.

**Motivación.** Conseguir pares documento-resumen anotados es costoso. El aporte práctico del artículo es reducir esa necesidad: con 1.000 ejemplos de ajuste supera el estado del arte previo en 6 de 12 conjuntos.

**Objetivo.** Implementar la inferencia de una arquitectura Transformer encoder-decoder, desplegarla en una interfaz y explicar su mecanismo de atención y la generación de Q, K y V.

---

## 3. Marco teórico

### Arquitectura

PEGASUS usa el Transformer encoder-decoder de Vaswani et al. (2017) **sin cambios estructurales**. Valores leídos del checkpoint:

| Componente | Valor | | Componente | Valor |
|---|---|---|---|---|
| Capas encoder | 16 | | Feed-forward | 4096 |
| Capas decoder | 16 | | Vocabulario | 96.103 |
| d_model | 1024 | | Entrada máxima | 1024 tokens |
| Cabezas | 16 | | Parámetros | 571 M |
| d_k | 64 | | | |

Cada capa del encoder tiene dos subcapas (autoatención y feed-forward); cada capa del decoder tiene tres, porque añade la atención cruzada.

### Mecanismo de atención

| Tipo | Ubicación | Q viene de | K y V vienen de |
|---|---|---|---|
| Autoatención | Encoder | documento | documento |
| Autoatención enmascarada | Decoder | resumen parcial | resumen parcial |
| Atención cruzada | Decoder | resumen parcial | salida del encoder |

En la **atención cruzada**, que es la relevante para resumir:

- **Q (query)** sale del resumen en construcción: *«¿qué necesito del documento para la siguiente palabra?»*
- **K (key)** sale del documento: la etiqueta con la que cada token se anuncia.
- **V (value)** sale del documento: el contenido que aporta si resulta elegido.

Las tres se obtienen multiplicando por matrices aprendidas distintas (`q_proj`, `k_proj`, `v_proj`, de 1024×1024), repartidas en 16 cabezas de 64 dimensiones. Luego se aplica:

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
```

con `d_k = 64` y escala `1/√64 = 0,125`. La división evita que el softmax se sature.

### Innovación

La arquitectura no es la innovación. El aporte es el **objetivo de preentrenamiento**, *Gap Sentences Generation*: en lugar de enmascarar palabras sueltas como BERT, se eliminan **oraciones completas** —las más importantes, medidas por su ROUGE1-F1 contra el resto del documento— y se entrena al modelo a generarlas. Como esa tarea se parece a resumir, el modelo aprende mejor y necesita menos datos etiquetados.

---

## 4. Metodología

No se entrena desde cero: se usan los **pesos preentrenados** publicados por los autores y se implementa solo la inferencia.

| Herramienta | Versión |
|---|---|
| Python | 3.14 |
| PyTorch | 2.13 |
| Transformers | 4.49 |
| SentencePiece | 0.2.2 |
| Streamlit | 1.64 |
| Matplotlib | 3.11 |

Ejecución local sobre CPU. El checkpoint pesa 2,2 GB y queda en caché tras la primera descarga.

**Estructura del código**

```
├── nucleo.py           # Carga del modelo, inferencia y cálculo de Q, K y V
├── app.py    # Interfaz web 
└── capturas/
```

La lógica del modelo está aislada en `nucleo.py`, así que ambas interfaces usan el mismo código de inferencia.

---

## 5. Desarrollo e implementación

### Ejecución

```bash
pip install -r requirements.txt
streamlit run app.py      # http://localhost:8501 
```

### Carga de pesos

```python
tok = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
mod = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")
mod.eval()
```

En Streamlit se envuelve en `@st.cache_resource` para que el modelo no se recargue en cada interacción.

### Preprocesamiento

El tokenizador SentencePiece convierte el texto en dos tensores: `input_ids` (1, 148) con los identificadores del vocabulario, y `attention_mask` (1, 148) que marca qué posiciones son texto real.

### Inferencia

```python
generados = mod.generate(**entradas, num_beams=4, length_penalty=0.8, max_length=128)
```

El decoder genera de forma autorregresiva: en cada paso calcula una distribución sobre los 96.103 tokens del vocabulario y el token elegido condiciona el siguiente. Con `num_beams=4` mantiene las cuatro secuencias parciales más probables.

### Extracción de Q, K y V

PEGASUS usa **normalización previa**, por lo que el estado que recibe el módulo de atención cruzada no es el que entra a la capa. Tomar el tensor equivocado produce un error de 1,0. La solución fue capturar los tensores exactos con *forward hooks* sobre el módulo de atención, y con ellos reproducir la fórmula.

---

## 6. Resultados y análisis

### Caso de prueba

**Entrada** (131 palabras, 148 tokens): noticia de la OMS sobre esperanza de vida global.

**Salida** (68 palabras, 81 tokens):

> The World Health Organization announced that global life expectancy has increased by more than six years since the year 2000 . The agency attributed the improvement to better access to vaccines, cleaner drinking water and reductions in child mortality across low-income countries . However, the report warned that progress has slowed considerably since 2019, when the COVID-19 pandemic reversed several health gains and overwhelmed hospital systems worldwide .

### Métricas

| Métrica | Valor |
|---|---|
| Tokens entrada / salida | 148 / 81 |
| Compresión | 1,8x |
| Tiempo (CPU) | 4,5 s |

### Formas de los tensores

| Tensor | Forma | Origen |
|---|---|---|
| Q | (1, 16, 80, 64) | 80 posiciones del resumen |
| K, V | (1, 16, 148, 64) | 148 tokens del documento |
| Pesos | (1, 16, 80, 148) | cada palabra del resumen sobre cada token |

### Verificación

| | |
|---|---|
| Diferencia cálculo manual vs. modelo | **0.0** |
| Capas verificadas | 16 de 16 |
| Suma de cada fila de pesos | 1,000000 |

### Capturas

**Figura 1. Entrada**
![Entrada](images/entrada.png)

**Figura 2. Salida, métricas y formas de Q, K y V**
![Salida](images/salida-qkv.png)

**Figura 3. Verificación y mapa de atención cruzada**
![Atención](images/mapa.png)

### Análisis

**La salida es compresiva más que reescritura.** El mecanismo es abstractivo —el decoder genera libremente sobre todo el vocabulario— pero de las cinco oraciones del documento el modelo conservó tres y solo borró elementos accesorios. Es coherente con el checkpoint: CNN/DailyMail contiene resúmenes bastante extractivos y el modelo aprendió ese estilo. Un checkpoint ajustado sobre XSum reescribiría mucho más.

**Attention sink.** Al promediar los pesos sobre las 16 cabezas, la atención se concentra en tokens poco informativos (preposiciones, puntuación) con pesos de 0,2 a 0,5. Por eso el mapa se muestra por cabeza individual: el promedio oculta la alineación real.

**Fidelidad.** No se observaron datos inventados: todas las cifras del resumen aparecen en el documento.

---

## 7. Conclusiones

**Aprendizajes.** El aporte de PEGASUS no está en la arquitectura sino en el diseño del objetivo de preentrenamiento. En la implementación, el aprendizaje principal fue que entender la teoría no equivale a poder reproducirla: el primer cálculo de Q·K daba error 1,0 por tomar el estado de entrada de la capa, que es lo que sugiere el diagrama clásico. Solo al advertir la normalización previa y usar hooks el cálculo coincidió con el modelo.

**Limitaciones.**

1. Los checkpoints son de inglés; con español la salida es incoherente.
2. La entrada máxima es de 1.024 tokens y los documentos largos se truncan.
3. El ajuste sobre CNN/DailyMail sesga la salida hacia la compresión.
4. No se calcularon métricas ROUGE sobre un conjunto de prueba completo.

**Mejoras posibles.**

- Evaluar con ROUGE sobre el test de CNN/DailyMail y comparar con los 44,17 / 21,47 / 41,11 del artículo.
- Añadir el baseline Lead-3 y medir n-gramas novedosos para cuantificar la abstracción.
- Ajuste fino con corpus en español.
- Visualizar también la autoatención del encoder y del decoder.

---

## 8. Referencias

[1] J. Zhang, Y. Zhao, M. Saleh y P. J. Liu, «PEGASUS: Pre-training with extracted gap-sentences for abstractive summarization», en *Proc. 37th Int. Conf. Mach. Learn. (ICML)*, vol. 119, PMLR, 2020. [En línea]. Disponible en: https://arxiv.org/abs/1912.08777

[2] A. Vaswani *et al.*, «Attention is all you need», en *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 5998–6008.

[3] C.-Y. Lin, «ROUGE: A package for automatic evaluation of summaries», en *Text Summarization Branches Out*, Barcelona, España: ACL, 2004, pp. 74–81.

[4] Google Research, «PEGASUS», repositorio GitHub. [En línea]. Disponible en: https://github.com/google-research/pegasus
