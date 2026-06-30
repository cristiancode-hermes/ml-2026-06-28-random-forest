# 🌲 Random Forest — Predicción de Riesgo de Préstamos Bancarios

**Nivel:** 3 (ML clásico — scikit-learn) | **Skill:** Random Forest

---

## 🎯 Objetivo del proyecto

Aprenderás **Random Forest**, uno de los modelos de machine learning más potentes y populares. Verás por qué **100 árboles piensan mejor que 1**, cómo se comparan con un árbol de decisión simple, y cómo ajustar sus hiperparámetros.

Trabajaremos con un problema realista: **predecir si un cliente de banco va a pagar o no su préstamo**, usando datos sintéticos con reglas realistas.

---

## 📖 Explicación teórica

### ¿Qué es Random Forest?

Random Forest es un **ensamble** (conjunto) de muchos árboles de decisión. La idea es simple pero poderosa:

> **"Un solo experto puede equivocarse. 100 expertos votando rara vez se equivocan."**

### ¿Cómo funciona?

1. **Bootstrap**: Cada árbol se entrena con una muestra ALEATORIA de los datos (con reemplazo). Esto hace que cada árbol vea datos ligeramente diferentes.

2. **Random Subset of Features**: En cada división del árbol, solo se considera un subconjunto ALEATORIO de características. Esto evita que todos los árboles se enfoquen en las mismas variables.

3. **Agregación**: Para clasificación, cada árbol vota y gana la clase mayoritaria. Para regresión, se promedian las predicciones.

### ¿Por qué es mejor que un árbol solo?

| Problema | Árbol simple | Random Forest |
|----------|-------------|---------------|
| **Sobreajuste** | Alto — memoriza ruido | Bajo — el promedio cancela el ruido |
| **Varianza** | Alta — pequeños cambios en datos cambian mucho el árbol | Baja — el ensamble es estable |
| **Precisión** | Buena en train, peor en test | Excelente en ambos |
| **Interpretación** | Fácil — se puede visualizar | Más difícil — pero tenemos feature importance |

### Hiperparámetros clave

- **n_estimators**: número de árboles (más = mejor, pero más lento)
- **max_depth**: profundidad máxima de cada árbol
- **min_samples_split**: mínimo de muestras para dividir un nodo
- **max_features**: cuántas características considerar en cada división
- **oob_score**: Out-of-Bag score (evaluación sin necesidad de test set)

---

## 🧪 Código paso a paso

### 1. Generación de datos sintéticos realistas 🏦

Creamos 1,500 clientes bancarios con características como `credit_score`, `income`, `dti_ratio`, `employment_years`, etc. Luego definimos **reglas del mundo real** para determinar quién tiene más probabilidad de caer en default:

- Si `credit_score < 500`: +30% de riesgo
- Si `dti_ratio > 40`: +25% de riesgo
- Si ya tuvo impagos antes: +20% de riesgo
- Si tiene < 2 años de empleo: +15% de riesgo

```python
# Generamos cada característica con distribuciones realistas
credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
income = np.random.lognormal(mean=10.5, sigma=0.6, size=n_samples)
dti_ratio = np.random.beta(2, 5, n_samples) * 50  # Sesgada a la izquierda
```

### 2. Análisis Exploratorio 📊

Comparamos las distribuciones de cada característica entre clientes que pagaron y los que cayeron en default. Los boxplots muestran claras diferencias:

```python
# Boxplot comparativo: pagadores vs default
bp = ax.boxplot([
    df[df['loan_default'] == 0]['credit_score'],
    df[df['loan_default'] == 1]['credit_score']
], labels=['Pagó 👍', 'Default 👎'])
```

### 3. Modelo Base: Árbol de Decisión 🌳

Entrenamos un árbol de decisión con `max_depth=3` para tener un baseline de comparación:

```python
dt_base = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_base.fit(X_train, y_train)
```

### 4. Random Forest 🌲

Entrenamos un bosque de 100 árboles con los parámetros por defecto:

```python
rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
```

### 5. Feature Importance 🔍

Comparamos qué características considera importantes cada modelo:

```python
rf_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_base.feature_importances_
}).sort_values('importance', ascending=False)
```

### 6. Curvas ROC 📈

Comparamos la calidad de clasificación de ambos modelos:

```python
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
```

### 7. Ajuste de Hiperparámetros ⚙️

Probamos desde 1 hasta 500 árboles para ver cómo mejora el rendimiento:

```python
for n in [1, 5, 10, 25, 50, 100, 200, 500]:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)
```

---

## 📊 Resultados

### Comparación de Modelos

```
╔════════════════╦══════════════╦════════════════╦══════════╗
║    Métrica     🌳 Árbol Simple 🌲 Random Forest  Mejora   ║
╠════════════════╬══════════════╬════════════════╬══════════╣
║ Accuracy       │    0.82      │     0.88       │  +7.3%   ║
║ Precision      │    0.68      │     0.79       │ +16.2%   ║
║ Recall         │    0.59      │     0.67       │ +13.6%   ║
║ F1-Score       │    0.63      │     0.73       │ +15.9%   ║
║ AUC-ROC        │    0.82      │     0.91       │ +11.0%   ║
╚════════════════╩══════════════╩════════════════╩══════════╝
```

### Feature Importance (Top 5 - Random Forest)

1. **credit_score** — el factor más determinante (relacionado directamente con la capacidad de pago)
2. **dti_ratio** — relación deuda-ingreso (indica sobreendeudamiento)
3. **default_history** — historial de impagos previos
4. **employment_years** — estabilidad laboral
5. **income** — ingresos del cliente

### Validación Cruzada (5-fold)

- **Árbol Simple**: 0.82 ± 0.03 (mayor variabilidad, menos robusto)
- **Random Forest**: 0.87 ± 0.02 (menor variabilidad, más consistente)

### Curva de Aprendizaje

Con **solo 10 árboles** ya superamos al árbol simple. Con **50-100 árboles** llegamos al rendimiento máximo. A partir de ahí, más árboles dan mejoras marginales.

---

## 💡 Conceptos clave

- **Ensemble**: Combinar múltiples modelos débiles para crear uno fuerte
- **Bootstrap**: Muestreo con reemplazo para crear diversidad entre los árboles
- **Bagging**: Bootstrap + Aggregating — la base de Random Forest
- **Feature Importance**: Mide cuánto contribuye cada característica a las decisiones del modelo
- **Out-of-Bag (OOB)**: Datos que no se usaron para entrenar un árbol (1/3 del bootstrap), sirven como validación interna
- **Hiperparámetros**: Configuraciones que controlan cómo aprende el modelo (n_estimators, max_depth)
- **Curva ROC**: Visualiza el balance entre sensibilidad y especificidad
- **AUC**: Área bajo la curva ROC (1 = perfecto, 0.5 = aleatorio)

---

## 🚀 Siguiente paso

**Validación Cruzada y Métricas Avanzadas** — Aprenderás técnicas más sofisticadas de validación como Stratified K-Fold, Leave-One-Out, y métricas para problemas con datos desbalanceados. También verás curvas de aprendizaje para detectar sesgo vs varianza.

---

## 📂 Estructura del proyecto

```
2026-06-28-random-forest/
├── README.md           ← Este archivo (explicación completa)
├── project.py          ← Código principal con comentarios en español
├── requirements.txt    ← Dependencias
├── data/
│   ├── loan_data.csv           ← Dataset generado
│   └── metrics_summary.csv     ← Métricas comparativas
├── docs/               ← Visualizaciones (PNG)
│   ├── 01_eda_boxplots.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_decision_tree.png
│   ├── 04_feature_importance.png
│   ├── 05_confusion_matrices.png
│   ├── 06_roc_curves.png
│   └── 07_tuning_n_estimators.png
└── LEARNINGS.md        ← Reflexiones del proyecto
```
