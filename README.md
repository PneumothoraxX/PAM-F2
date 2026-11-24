# PAM-F2 — Modelos de Decisión de Paso Único para Bucles Agénticos

Implementación de la **Fase 2 (Entrega 2 / Entrega Final)** del proyecto de investigación del Área de IA de ACECOM (UNI): *Desacople de la Decisión en Bucles Agénticos: Modelos de Decisión de Paso Único como Alternativa a la Decodificación Autorregresiva*.

## Resumen de la Arquitectura

1. **Nivel 1 — Modelo de Decisión ($T = 1$):** Un modelo compacto (familia Qwen) adaptado con LoRA resuelve la selección de herramienta/intención en una sola pasada de *prefill* leyendo los logits en `<readout>` sobre un conjunto cerrado de tokens reservados (`<opt_0>` a `<opt_254>`), sin decodificación autorregresiva ni KV-cache dinámico.
2. **Calibración Post-Hoc:** Ajuste de temperatura $T^*$ mediante **L-BFGS** sobre validación para acotar el *Expected Calibration Error* ($\text{ECE} < 0.05$).
3. **Compuerta Estocástica Híbrida:** Si $\max_i \hat{q}_i \ge \tau$, se dispara la acción directamente; si $\max_i \hat{q}_i < \tau$, se escala la consulta a un LLM reflexivo de respaldo (**ReAct**).

## Organización del Proyecto

* **Fase Experimental 1 (`src/phase1_banking77/`):** Evaluación aislada del Nivel 1 + calibración L-BFGS + comparativa frente a ReAct sobre **BANKING77** (77 intenciones).
* **Fase Experimental 2 (`src/phase2_hybrid_bfcl/`):** Compuerta híbrida ($\tau$) + llamadas a APIs reales (**BFCL**) + *Tree Batching* *(próximo bloque)*.
* **Núcleo compartido (`src/core/`):** Tokens reservados, formateo de prompts $T=1$, modelo LoRA, calibración y baseline ReAct.

```text
PAM-F2/
├── configs/
├── notebooks/
├── results/
│   ├── phase1_banking77/
│   └── phase2_bfcl/
├── src/
│   ├── core/
│   │   └── prompts.py              # Tokens <opt_i>, <readout> y formateador T=1
│   └── phase1_banking77/
│       └── data.py                 # Splits train/val/test y Dataset de BANKING77
├── tests/
│   ├── test_core_prompts.py
│   └── test_phase1_banking77.py
├── requirements.txt
└── README.md
```

## Configuración del Entorno

Requiere **Python 3.11 o 3.12** (para compatibilidad con `torch` y CUDA).

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Uso Actual (Pipeline de Datos — Fase 1)

Para descargar **BANKING77** (`PolyAI/banking77`), generar los splits estratificados (`train` / `val` / `test`) y visualizar un prompt formateado con `<opt_k>` y `<readout>`:

```bash
python -m src.phase1_banking77.data
```

Para ejecutar toda la suite de pruebas:

```bash
python -m unittest discover -s tests -v
```
