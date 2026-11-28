# PAM-F2 — Modelos de Decisión de Paso Único para Bucles Agénticos

Implementación completa (**Fase 1 y Fase 2**) del proyecto de investigación del Área de IA de ACECOM (UNI): *Desacople de la Decisión en Bucles Agénticos: Modelos de Decisión de Paso Único como Alternativa a la Decodificación Autorregresiva*.

## Resumen de la Arquitectura

1. **Nivel 1 — Modelo de Decisión ($T = 1$):** Un modelo compacto (familia Qwen) adaptado con LoRA resuelve la selección de herramienta/intención en una sola pasada de *prefill* leyendo los logits en `<readout>` sobre un conjunto cerrado de tokens reservados (`<opt_0>` a `<opt_254>`), sin decodificación autorregresiva ni KV-cache dinámico.
2. **Calibración Post-Hoc:** Ajuste de temperatura $T^*$ mediante **L-BFGS** sobre validación para acotar el *Expected Calibration Error* ($\text{ECE} < 0.05$).
3. **Compuerta Estocástica Híbrida + Tree Batching:**
   * Si $\max_i \hat{q}_i \ge \tau$, se dispara la acción directamente desde el Nivel 1 (**0 % fallo JSON**); si $\max_i \hat{q}_i < \tau$, se escala la consulta a un LLM reflexivo de respaldo (**ReAct**).
   * **Tree Batching:** Empaqueta un estado compartido $s_t$ con $M$ preguntas concurrentes en una sola secuencia mediante una máscara de atención causal 2D/4D y `position_ids` por rama.

## Organización del Proyecto

```text
PAM-F2/
├── configs/
├── notebooks/
│   ├── 01_phase1_banking77_t4.ipynb # Notebook Fase 1 (Kaggle/Colab GPU T4)
│   └── 02_phase2_bfcl_t4.ipynb      # Notebook Fase 2 (Compuerta Híbrida + Tree Batching)
├── results/
│   ├── phase1_banking77/            # Métricas y 3 figuras de Fase 1 (BANKING77)
│   └── phase2_bfcl/                 # Métricas y 3 figuras de Fase 2 (BFCL + Tree Batching)
├── src/
│   ├── core/
│   │   ├── prompts.py               # Tokens <opt_i>, <readout> y formateador T=1
│   │   ├── model.py                 # DecisionModel (vocab base congelado + LoRA, paso único)
│   │   ├── calibration.py           # TemperatureScaler (L-BFGS), ECE y Reliability Diagrams
│   │   └── react_baseline.py        # Baseline ReAct (T>1) y benchmark en vivo T=1 vs T>1
│   ├── phase1_banking77/
│   │   ├── data.py                  # Splits estratificados train/val/test y Dataset
│   │   ├── train.py                 # Entrenamiento LoRA y exportación de logits
│   │   └── evaluate.py              # Calibración L-BFGS, métricas ECE y figuras Fase 1
│   └── phase2_hybrid_bfcl/
│       ├── data.py                  # Catálogo de 24 APIs estilo BFCL y escenarios multi-pregunta
│       ├── gate.py                  # Compuerta estocástica híbrida (τ) y Tree Batching 2D
│       └── evaluate.py              # Evaluación híbrida, Tree Batching y 3 figuras Fase 2
├── tests/
│   ├── test_core_prompts.py
│   ├── test_core_model.py
│   ├── test_core_calibration.py
│   ├── test_core_react.py
│   ├── test_phase1_banking77.py
│   ├── test_phase1_train.py
│   ├── test_phase1_evaluate.py
│   └── test_phase2_bfcl.py
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

## Ejecución y Pruebas

1. **Correr toda la suite de tests (23 pruebas unitarias y de integración):**
   ```bash
   PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
   ```
2. **Fase 1 (BANKING77 — Entrenamiento, Calibración L-BFGS y Benchmark vs. ReAct):**
   ```bash
   PYTHONPATH=. .venv/bin/python -m src.phase1_banking77.train --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3 --batch-size 8
   PYTHONPATH=. .venv/bin/python -m src.phase1_banking77.evaluate --run-live-benchmark --benchmark-samples 50
   ```
3. **Fase 2 (BFCL — Compuerta Estocástica Híbrida $\tau$ y Tree Batching 2D):**
   ```bash
   PYTHONPATH=. .venv/bin/python -m src.phase2_hybrid_bfcl.evaluate --tau 0.75
   ```
