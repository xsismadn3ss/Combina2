# Combina2 — evaluación E/G/C

- Generado: `2026-10-20T19:11:35.828384+00:00`
- Git: `bd0f5d0`
- Entorno: Python `3.12.14` · NumPy `2.5.4` · TF `2.21.0` · Keras `3.15.1` · sklearn `1.9.1`

## Modelos activos

| Slot | Archivo | sha256 | Resuelto | Estado |
|---|---|---|---|---|
| extract | `extract-v1.keras` | `051ce4a01abb2152…` | `extract-v1.keras` | OK |
| generate | `generate-v1.keras` | `1c91464fbeadcec2…` | `generate-v1.keras` | OK |
| classify | `classify-v2.keras` | `e04a77fb384eac5c…` | `classify-v2.keras` | OK |

## extract

| Métrica | Valor | Umbral | Estado |
|---|---|---|---|
| `delta_e76_mean` | 3.8119 | <= 5.0000 | **PASS** |
| `delta_e76_max` | 14.1106 | <= 20.0000 | **PASS** |

- ΔE76 medio/máx: `3.8119` / `14.1106`; fallbacks: `0/10`

## generate

| Métrica | Valor | Umbral | Estado |
|---|---|---|---|
| `ciede2000_mean` | 13.0981 | <= 14.5000 | **PASS** |
| `mse` | 0.0256 | <= 0.0500 | **PASS** |

- CIEDE2000 medio `13.0981` · CIE76 medio `23.5952` · MSE `0.025599`
- Score: media `0.9423`, std `0.0319`, rango [`0.8088`, `0.9764`]
- AUC(score->buena reconstrucción, CIE76 <= 14.0) = `0.3516`
- Calibración: **no aplicada** — motivo `score_auc_below_threshold` (AUC mínimo 0.55; rechazada sin fallar el gate).

## classify

| Métrica | Valor | Umbral | Estado |
|---|---|---|---|
| `accuracy` | 0.9750 | >= 0.9000 | **PASS** |
| `macro_f1` | 0.9757 | >= 0.9000 | **PASS** |
| `recall/complementario` | 0.9474 | >= 0.8500 | **PASS** |
| `recall/triada` | 0.9565 | >= 0.8500 | **PASS** |
| `recall/analogo` | 1.0000 | >= 0.8500 | **PASS** |
| `recall/monocromatica` | 1.0000 | >= 0.8500 | **PASS** |
| `calibration/ece` | 0.0275 | <= 0.1000 | **PASS** |
| `calibration/nll` | 0.0929 | <= 0.1324 | **PASS** |

- accuracy `0.9750` · macro-F1 `0.9757` · NLL `0.1324` · ECE `0.0916`
- Recall por clase: complementario `0.9474`, triada `0.9565`, analogo `1.0000`, monocromatica `1.0000`
- Confusión (filas = verdad):

```
[18,  0,  0,  1]
[ 1, 22,  0,  0]
[ 0,  0, 16,  0]
[ 0,  0,  0, 22]
```
- Medición en test: NLL cruda `0.1324`, ECE cruda `0.0916`; T óptima (solo medición) `0.48` -> NLL `0.0717`, ECE `0.0292`
- Calibración APLICADA (T ajustada en validación): T = `0.266489`; ECE `0.0916` -> `0.0275`, NLL `0.1324` -> `0.0929` (gate ECE ≤ `0.10` y NLL no peor).

## Manifiesto

OK: 26 métricas comparadas (tolerancia 10%).

## Resumen

- Gates: 12/12 PASS
- Exit code: `0`

