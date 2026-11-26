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
│   └── 01_phase1_banking77_t4.ipynb # Notebook listo para Kaggle/Colab (GPU T4)
├── results/
│   ├── phase1_banking77/
│   └── phase2_bfcl/
├── src/
│   ├── core/
│   │   ├── prompts.py              # Tokens <opt_i>, <readout> y formateador T=1
│   │   ├── model.py                # DecisionModel (base_model + LoRA, paso único)
│   │   ├── calibration.py          # TemperatureScaler (L-BFGS), ECE y Reliability Diagrams
│   │   └── react_baseline.py       # Baseline ReAct (T>1) y benchmark en vivo T=1 vs T>1
│   └── phase1_banking77/
│       ├── data.py                 # Splits estratificados train/val/test y Dataset
│       ├── train.py                # Entrenamiento LoRA y exportación de logits
│       └── evaluate.py             # Calibración L-BFGS, métricas ECE y figuras PNG
├── tests/
│   ├── test_core_prompts.py
│   ├── test_core_model.py
│   ├── test_core_calibration.py
│   ├── test_core_react.py
│   ├── test_phase1_banking77.py
│   ├── test_phase1_train.py
│   └── test_phase1_evaluate.py
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

## Ejecución de Fase 1 y Pruebas

1. **Correr toda la suite de tests (19 pruebas):**
   ```bash
   PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
   ```
2. **Previsualizar el dataset BANKING77 y formato de prompt ($T=1$):**
   ```bash
   PYTHONPATH=. .venv/bin/python -m src.phase1_banking77.data
   ```
3. **Entrenar `DecisionModel` con LoRA (en GPU T4 / Kaggle):**
   ```bash
   PYTHONPATH=. .venv/bin/python -m src.phase1_banking77.train --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3
   ```
4. **Calibrar con L-BFGS, correr benchmark en vivo contra ReAct y generar gráficas:**
   ```bash
   PYTHONPATH=. .venv/bin/python -m src.phase1_banking77.evaluate --run-live-benchmark --benchmark-samples 50
   ```
   *(Para probar localmente la calibración y generación de las 3 gráficas sin haber corrido el entrenamiento aún, usa `--demo-synthetic`).*
