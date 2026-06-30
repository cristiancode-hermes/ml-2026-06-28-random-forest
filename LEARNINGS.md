# Learnings — 2026-06-28

## Qué se hizo
Proyecto de Random Forest para clasificación de riesgo de préstamos bancarios.
Se generaron datos sintéticos realistas (1,500 clientes, 10 características),
se comparó un Árbol de Decisión simple con un Random Forest de 100 árboles,
y se visualizaron métricas, feature importance, curvas ROC y ajuste de
hiperparámetros.

## Conceptos nuevos
- [x] Random Forest: qué es y cómo funciona (Bootstrap + Agregación)
- [x] Bagging (Bootstrap Aggregating) — por qué reduce la varianza
- [x] Random Subset of Features — por qué es necesario
- [x] Feature Importance de Random Forest (más robusta que DT sola)
- [x] Comparación sistemática: Árbol Simple vs Random Forest
- [x] Curva ROC y AUC para clasificación binaria
- [x] Validación Cruzada (5-fold) para medir consistencia
- [x] Ajuste de n_estimators (curva de aprendizaje)
- [x] Grid Search para max_depth
- [ ] n_estimators óptimo: ~100 árboles es suficiente

## Dificultades encontradas
- Ninguna técnica relevante. El código corrió sin problemas una vez corregido
  el error de sintaxis (keyword arg después de positional arg en lognormal).

## Notas para el próximo proyecto
- Pasar a Validación Cruzada y Métricas Avanzadas
- Cubrir: K-Fold, Stratified K-Fold, Leave-One-Out, curvas de aprendizaje,
  curvas de validación, y métricas para clasificación desbalanceada
- Usar el mismo dataset de préstamos o uno similar para comparar
