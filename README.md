[readme_md.md](https://github.com/user-attachments/files/34602916/readme_md.md)
# Experimento 12: Hyperparameter Tuning — Optimización Bayesiana

> **¿Es funcional una Optimización Bayesiana que intente optimizar más hiperparámetros al mismo tiempo?**

---

## 🏛️ Información Institucional y Equipo

* **Institución:** Universidad Austral — Facultad de Ingeniería
* **Programa:** Maestría en Ciencia de Datos (Online) — Cohorte 2026-2027
* **Materia:** Laboratorio I
* **Docentes:** Prof. Gustavo Denicolay | Prof. Fernando Arrieta
* **Equipo (Grupo B):**
  * Chavez Ortega, Delfina
  * Pena, Carlos

---

## 🔬 Hipótesis Experimental (Problema #12)

> **Hipótesis del Grupo B:**  
> *"A medida que se incrementa el espacio de búsqueda con nuevos hiperparámetros en la Optimización Bayesiana, se produce una mejora estadísticamente significativa de la ganancia del conjunto de testing (Julio 2021) respecto al baseline."*


---

## 🧠 Sesgos Cognitivos Identificados

* **Delfina Chavez:** Riesgo de *anclaje* al utilizar los rangos teóricos de la bibliografía como verdades absolutas, asumiendo que los límites propuestos serán óptimos para la distribución particular de los datos de este laboratorio.
* **Carlos Pena:** Creencia de que optimizar una mayor cantidad de hiperparámetros (HPs) permitirá obtener sistemáticamente una mayor ganancia, asumiendo que el costo computacional y de tiempo asociado será elevado.

---

## 📐 Diseño Experimental

Se diseñaron tres (3) configuraciones principales de experimentos para evaluar el impacto de la cantidad de HPs optimizados mediante Optimización Bayesiana (300 iteraciones de BO por experimento con 10 semillas distintas):

1. **Experimento A (Baseline - 5 HPs):** Hiperparámetros centrales (*Core*), de control de aprendizaje y regularización (`num_iterations`, `num_leaves`, `learning_rate`, `feature_fraction`, `min_sum_hessian_in_leaf`).
2. **Experimento B (11 HPs):** Agrega a los 5 HPs anteriores otros 6 enfocados en regularización, complejidad del árbol y muestreo (`max_depth`, `min_data_in_leaf`, `lambda_l1`, `lambda_l2`, `bagging_fraction`, `bagging_freq`).
3. **Experimento C (17 HPs Teóricos / Adaptaciones):** Agrega balance de clases, cuantización de gradientes y selección por nodo (`scale_pos_weight`, `min_gain_to_split`, `use_quantized_grad`, `num_grad_quant_bins`, `stochastic_rounding`, `feature_fraction_bynode`).

---

## 🛠️ Incidente Técnico, Debugging y Adaptación (Experimento C)

### 🚨 El Problema
Al ejecutar el **Experimento C (17 HPs)**, la librería `mlrMBO` colapsó con un error fatal en LightGBM:
```text
Check failed: (best_split_info.left_count) > (0)
```

### 🔍 Causa Raíz
Se identificó que la **cuantización de gradientes** (`use_quantized_grad = TRUE`) generaba un colapso matemático interno en C++ al interactuar conjuntamente con `num_grad_quant_bins` y `stochastic_rounding = TRUE`.

### 🛠️ Resolución y Diseños de Contingencia
Tras 6 horas de debugging manual con impresiones de control previas a la llamada a LightGBM:
1. **Exp. C Alternativo (14 HPs - "El Seguro"):** Excluye los 3 HPs conflictivos de cuantización manteniendo la comparabilidad estricta de los rangos originales.
2. **Exp. C Modificado (16 HPs - "El Domado"):** Desactiva `stochastic_rounding = FALSE` y reajusta los rangos de `num_grad_quant_bins` ($[512, 1024]$) y `min_data_in_leaf` ($[300, 500]$) para evitar la partición vacía en C++.

---

## 🎛️ Definición de Rangos de Hiperparámetros

| Hiperparámetro | Exp. A (5 HP) y B (11 HP) | Exp. C Diseñado (17 HP) | Exp. C Alternativo (14 HP) | Exp. C Real (16 HP) |
| :--- | :---: | :---: | :---: | :---: |
| `num_iterations` | $[4, 1024]$ | $[4, 1024]$ | $[4, 1024]$ | $[4, 1024]$ |
| `num_leaves` | $[4, 2048]$ | $[4, 2048]$ | $[4, 2048]$ | $[4, 2048]$ |
| `learning_rate` | $[0.0078125, 0.933033]$ | $[0.0078125, 0.933033]$ | $[0.0078125, 0.933033]$ | $[0.0078125, 0.933033]$ |
| `feature_fraction` | $[0.05, 0.90]$ | $[0.05, 0.90]$ | $[0.05, 0.90]$ | $[0.05, 0.90]$ |
| `min_sum_hessian_in_leaf` | $[0.0009765625, 1024]$ | $[0.0009765625, 1024]$ | $[0.0009765625, 1024]$ | $[0.0000009536, 1024]$ |
| `max_depth` | — / $[ -1, 3:15 ]$ | $[-1, 3:15]$ | $[-1, 3:15]$ | $[-1, 3:15]$ |
| `min_data_in_leaf` | — / $[5, 300]$ | $[5, 300]$ | $[5, 300]$ | $[300, 500]$ |
| `lambda_l1` | — / $[1\text{E-}9, 100]$ | $[1\text{E-}9, 100]$ | $[1\text{E-}9, 100]$ | $[1\text{E-}9, 100]$ |
| `lambda_l2` | — / $[1\text{E-}9, 100]$ | $[1\text{E-}9, 100]$ | $[1\text{E-}9, 100]$ | $[1\text{E-}9, 100]$ |
| `bagging_fraction` | — / $[0.4, 1]$ | $[0.4, 1]$ | $[0.4, 1]$ | $[0.95, 1]$ |
| `bagging_freq` | — / $[0, 5]$ | $[0, 5]$ | $[0, 5]$ | $[0, 5]$ |
| `scale_pos_weight` | — | $[0.01, 10]$ | $[0.01, 10]$ | $[0.01, 10]$ |
| `min_gain_to_split` | — | $[0.00001, 2]$ | $[0.00001, 2]$ | $[0.00001, 0.01]$ |
| `feature_fraction_bynode` | — | $[0.4, 1]$ | $[0.4, 1]$ | $[0.4, 1]$ |
| `use_quantized_grad` | — | `TRUE` / `FALSE` | *Excluido* | `TRUE` / `FALSE` |
| `num_grad_quant_bins` | — | $[4, 32]$ | *Excluido* | $[512, 1024]$ |
| `stochastic_rounding` | — | `TRUE` / `FALSE` | *Excluido* | `FALSE` |

---

## 📊 Tabla de Resultados (Ganancia sobre Testing por Semilla)

| Semilla | Exp. A (5 HP) | Exp. B (11 HP) | Exp. C (16 HP Mod.) | Exp. C (14 HP Alt.) |
| :---: | :---: | :---: | :---: | :---: |
| **179213** | 95.51 | 92.30 | 92.43 | 92.19 |
| **200017** | 92.23 | 89.58 | 89.71 | 92.53 |
| **323503** | 92.20 | 96.55 | 90.66 | 88.49 |
| **543121** | 95.65 | 93.08 | 89.45 | 89.85 |
| **607681** | 94.97 | 93.03 | 88.77 | 91.33 |
| **611147** | 93.36 | 92.41 | 91.06 | 90.30 |
| **618271** | 90.02 | 89.47 | 95.17 | 93.85 |
| **726769** | 90.62 | 91.56 | 92.85 | 93.25 |
| **897563** | 94.00 | 91.00 | 91.39 | 91.55 |
| **898117** | 92.29 | 90.96 | 92.97 | 94.01 |

---

## 🧪 Pruebas de Hipótesis (Test de Wilcoxon)

Se aplicó la prueba no paramétrica de Wilcoxon ($N=10$) para evaluar si existían diferencias estadísticamente significativas ($\alpha = 0.05$):

| Comparación | Pregunta Experimental | P-Valor ($N=10$) | ¿Diferencia Significativa? | 
| :--- | :--- | :---: | :---: | :---: |
| **Exp. A vs Exp. B** | ¿Aumentar de 5 a 11 HPs suma valor? | `0.1309` | No | 
| **Exp. A vs Exp. C (14 HP)** | ¿Pasar de 5 a 14 HPs (Seguro) suma valor? | `0.2754` | No | 
| **Exp. A vs Exp. C (16 HP)** | ¿Pasar de 5 a 16 HPs (Cuantización) suma valor? | `0.1309` | No | 
| **Exp. B vs Exp. C (14 HP)** | ¿Balance/nodo suma valor sobre Exp. B? | `1.0000` | No | 
| **Exp. B vs Exp. C (16 HP)** | ¿Vale la pena la cuantización frente a B? | `0.8457` | No | 
| **Exp. C (14 HP) vs Exp. C (16 HP)** | ¿Cuantización vs. Alternativo Seguro? | `0.6250` | No | 

---

## 🏆 Conclusión Final

> **Conclusión:** **Con 5 HPs alcanza.**  
> No se encontró evidencia estadísticamente significativa de que expandir el espacio de búsqueda hacia 11, 14 o 16 hiperparámetros incremente la ganancia sobre el conjunto de testing respecto al *baseline*. Además, la incorporación de parámetros complejos (como cuantización de gradientes) incrementa la inestabilidad del algoritmo sin aportar mejoras en el rendimiento.

---

## 📁 Estructura de Scripts y Archivos del Repositorio

* **`Lanzador_semilla-EXP.ipynb`**: Script automatizado que ejecuta en bucle los experimentos llamando a los scripts individuales con las semillas configuradas.
* **`Exp_A_z621_Workflow_01_junior_GrupoB_BO.ipynb`**: Código de corrida para el Experimento A (5 HPs).
* **`Exp_B_z621_Workflow_01_junior_GrupoB_BO.ipynb`**: Código de corrida para el Experimento B (11 HPs).
* **`Exp_C_z621_Workflow_01_junior_GrupoB_BO_14HP.ipynb`**: Código del Experimento C Alternativo ("Seguro", 14 HPs).
* **`Exp_C_z621_Workflow_01_junior_GrupoB_BO_16HP.ipynb`**: Código del Experimento C Modificado ("Domado", 16 HPs con rangos ajustados).
* **`Debugging.ipynb`**: Script independiente utilizado para realizar el diagnóstico y la prueba aislada de rangos de LightGBM tras los errores en la librería C++.
* **`Test_Wilcoxon_Grupo_B.ipynb`**: Script de análisis estadístico para ejecutar el Test de Wilcoxon entre las corridas.
* **`ExpA.xlsx`**, **`ExpB.xlsx`**, **`ExpC-14HP-Alternativo.xlsx`**, **`ExpC-16HP.xlsx`**: Planillas con los registros completos de resultados.
* **`Resultados_Estadisticos_Definitivos.xlsx`**: Detalle del análisis estadístico y P-valores por semilla.
* **`Prueba_HP_Debugging.xlsx`**: Bitácora de pruebas y errores documentados durante la etapa de debugging.

---
*Universidad Austral — Facultad de Ingeniería — Maestría en Ciencia de Datos*
