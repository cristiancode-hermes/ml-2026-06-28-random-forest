#!/usr/bin/env python3
"""
==============================================================================
🌲 RANDOM FOREST — El Bosque Mágico de Árboles
Nivel 3 | ML clásico (scikit-learn) | Skill: Random Forest

Hoy vamos a aprender qué es Random Forest y por qué es mejor que un solo
árbol de decisión. Usaremos datos de préstamos bancarios para predecir
si un cliente va a pagar o no su préstamo.

📖 Concepto clave:
Un solo árbol de decisión es como preguntarle a UN experto. Random Forest
es como preguntar a 100 expertos y votar. El resultado es mucho más robusto
porque el promedio de muchos árboles reduce el error y el sobreajuste.

Autor: Hermes ML Teacher
Fecha: 2026-06-28
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Modo no interactivo para servidor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
# Para guardar nuestras visualizaciones
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paleta de colores profesional para los gráficos
COLOR_PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2E']
sns.set_palette(COLOR_PALETTE)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

np.random.seed(42)  # Para que los resultados sean reproducibles siempre

print("=" * 72)
print("  🌲 RANDOM FOREST — PREDICCIÓN DE RIESGO DE PRÉSTAMOS")
print("  Aprendiendo por qué 100 árboles piensan mejor que 1")
print("=" * 72)

# =============================================================================
# PASO 1: CREAMOS DATOS SINTÉTICOS REALISTAS 🏦
# =============================================================================
print("\n" + "=" * 72)
print("📌 PASO 1: Generando datos de préstamos bancarios...")
print("=" * 72)

# En la vida real, un banco tiene miles de clientes con diferentes perfiles.
# Vamos a crear 1500 clientes con características realistas.

n_samples = 1500

# --- Creamos las características una por una ---
# credit_score: puntaje crediticio (300-850), distribución normal centrada en 650
credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)

# income: ingreso anual, distribución log-normal (como los ingresos reales)
income = np.random.lognormal(mean=10.5, sigma=0.6, size=n_samples)

# loan_amount: monto del préstamo entre $1,000 y $50,000
loan_amount = np.random.uniform(1000, 50000, n_samples)

# loan_term: plazo del préstamo (12, 24, 36, 48 o 60 meses)
loan_term = np.random.choice([12, 24, 36, 48, 60], n_samples, p=[0.1, 0.15, 0.25, 0.25, 0.25])

# dti_ratio: relación deuda-ingreso (0-50%), distribución beta sesgada a la izquierda
dti_ratio = np.random.beta(2, 5, n_samples) * 50

# employment_years: años de empleo, distribución exponencial (más gente con pocos años)
employment_years = np.random.exponential(8, n_samples).clip(0, 45).round(1)

# num_credit_lines: número de líneas de crédito
num_credit_lines = np.random.poisson(3, n_samples) + 1

# default_history: si ha tenido impagos antes (30% sí)
default_history = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

# age: edad del cliente
age = np.random.normal(38, 12, n_samples).clip(18, 75).round(0).astype(int)

# education: nivel educativo (categórica)
education = np.random.choice(
    ['high_school', 'bachelor', 'master', 'phd'],
    n_samples, p=[0.3, 0.4, 0.2, 0.1]
)

# --- Armamos el DataFrame ---
df = pd.DataFrame({
    'credit_score': credit_score.round(0).astype(int),
    'income': income.round(0).astype(int),
    'loan_amount': loan_amount.round(0).astype(int),
    'loan_term': loan_term,
    'dti_ratio': dti_ratio.round(2),
    'employment_years': employment_years,
    'num_credit_lines': num_credit_lines,
    'default_history': default_history,
    'age': age,
    'education': education
})

print(f"✓ Dataset creado: {df.shape[0]} clientes, {df.shape[1]} características")
print(f"  Columnas: {', '.join(df.columns)}")
print(f"\n  Primeros 5 clientes:")
print(df.head().to_string())
print(f"\n  Información del dataset:")
print(df.info())

# =============================================================================
# PASO 2: CREAMOS LA VARIABLE OBJETIVO (TARGET) 🎯
# =============================================================================
print("\n" + "=" * 72)
print("📌 PASO 2: Generando la variable objetivo (loan_default)...")
print("=" * 72)

# Aquí está la magia: definimos REGLAS del mundo real para decidir quién
# cae en default. Así el modelo tendrá una señal real que aprender.
#
# Factores que aumentan el riesgo:
# - Credit score BAJO → más riesgo (el más importante)
# - DTI ratio ALTO → más riesgo
# - Default history = 1 (ya tuvo impagos) → más riesgo
# - Pocos años de empleo → más riesgo
# - Préstamo grande relativo al ingreso → más riesgo

def generate_default_probability(row):
    """Calcula la probabilidad de default según reglas realistas."""
    prob = 0.10  # Probabilidad base: 10%
    
    # --- Credit score (el factor MÁS importante) ---
    if row['credit_score'] < 500:
        prob += 0.30  # Crédito muy bajo → +30% riesgo
    elif row['credit_score'] < 600:
        prob += 0.15  # Crédito bajo → +15% riesgo
    elif row['credit_score'] < 700:
        prob += 0.05  # Crédito regular → +5% riesgo
    # 700+ → no añade riesgo adicional
    
    # --- Debt-to-Income ratio ---
    if row['dti_ratio'] > 40:
        prob += 0.25  # Demasiada deuda → +25%
    elif row['dti_ratio'] > 30:
        prob += 0.12
    elif row['dti_ratio'] > 20:
        prob += 0.05
    
    # --- Default history (historial de impagos) ---
    if row['default_history'] == 1:
        prob += 0.20  # Ya falló antes → +20%
    
    # --- Años de empleo (estabilidad laboral) ---
    if row['employment_years'] < 2:
        prob += 0.15  # Muy inestable → +15%
    elif row['employment_years'] < 5:
        prob += 0.05
    
    # --- Loan-to-income ratio ---
    if row['loan_amount'] > row['income'] * 0.5:
        prob += 0.10  # Préstamo muy grande relativo al ingreso
    
    return min(prob, 0.95)  # Limitamos al 95% máximo

# Aplicamos la función a cada fila
df['default_prob'] = df.apply(generate_default_probability, axis=1)

# Generamos la variable binaria (0 = paga, 1 = no paga)
df['loan_default'] = (np.random.random(n_samples) < df['default_prob']).astype(int)

# Eliminamos la columna auxiliar de probabilidad
df = df.drop('default_prob', axis=1)

# Mostramos la distribución de la variable objetivo
default_count = df['loan_default'].sum()
total_count = len(df)
print(f"\n  📊 Clientes que pagaron (0): {total_count - default_count} ({(1-default_count/total_count)*100:.1f}%)")
print(f"  📊 Clientes en default (1): {default_count} ({default_count/total_count*100:.1f}%)")
print(f"  ⚖️  Ratio: {default_count/(total_count-default_count):.2f} : 1 (default:no default)")
print(f"  ✓ Target desbalanceado pero manejable (común en datos reales)")

# =============================================================================
# PASO 3: ANÁLISIS EXPLORATORIO RÁPIDO 📊
# =============================================================================
print("\n" + "=" * 72)
print("📌 PASO 3: Análisis exploratorio — ¿cómo se diferencian los clientes?")
print("=" * 72)

# --- 3a: Boxplots comparativos ---
print("\n   Creando boxplots comparativos...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Comparación de características: Pagadores vs Default', fontsize=16, fontweight='bold')

features_to_plot = ['credit_score', 'dti_ratio', 'income', 'loan_amount', 'employment_years', 'age']
titles = ['Credit Score', 'DTI Ratio (%)', 'Ingreso Anual ($)', 'Monto del Préstamo ($)', 'Años de Empleo', 'Edad']

for idx, (feat, title) in enumerate(zip(features_to_plot, titles)):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    
    # Boxplot comparativo: 0 (pagó) vs 1 (default)
    bp = ax.boxplot([df[df['loan_default'] == 0][feat], df[df['loan_default'] == 1][feat]],
                     patch_artist=True)

    # Colores para cada caja
    bp['boxes'][0].set_facecolor('#2ECC71')
    bp['boxes'][1].set_facecolor('#E74C3C')
    ax.set_xticklabels(['Pagó 👍', 'Default 👎'])
    
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(feat)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_eda_boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ docs/01_eda_boxplots.png — Boxplots comparativos guardados")

# --- 3b: Matriz de correlación ---
print("\n   Calculando matriz de correlación...")
df_encoded = df.copy()
df_encoded['education_encoded'] = LabelEncoder().fit_transform(df['education'])
corr_matrix = df_encoded.select_dtypes(include=[np.number]).corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Matriz de Correlación — Datos de Préstamos', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ docs/02_correlation_heatmap.png — Heatmap de correlaciones")

# Observación rápida
corr_with_target = corr_matrix['loan_default'].drop('loan_default').sort_values(ascending=False)
print(f"\n  📈 Características más correlacionadas con default:")
for feat, corr_val in corr_with_target.head(5).items():
    direction = "👍 menos riesgo" if corr_val < 0 else "👎 más riesgo"
    print(f"     {feat:20s}: {corr_val:+.3f} ({direction})")

# =============================================================================
# PASO 4: PREPARAMOS LOS DATOS PARA EL MODELO 🔧
# =============================================================================
print("\n" + "=" * 72)
print("📌 PASO 4: Preparando datos para ML...")
print("=" * 72)

# Codificamos la variable categórica 'education'
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])
print(f"  ✓ Education codificada: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Separamos características (X) y objetivo (y)
feature_columns = ['credit_score', 'income', 'loan_amount', 'loan_term',
                   'dti_ratio', 'employment_years', 'num_credit_lines',
                   'default_history', 'age', 'education_encoded']

X = df[feature_columns].values
y = df['loan_default'].values

print(f"  ✓ X (features): {X.shape} — {X.shape[1]} características para {X.shape[0]} muestras")
print(f"  ✓ y (target): {y.shape} — {y.sum()} defaults, {len(y)-y.sum()} no-defaults")

# --- Train/Test Split ---
# 70% entrenamiento, 30% prueba
# stratify=y asegura que la proporción de defaults sea igual en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n  📊 Split:")
print(f"     Train: {X_train.shape[0]} muestras ({y_train.sum()} defaults, {(1-y_train.mean())*100:.0f}% pagadores)")
print(f"     Test:  {X_test.shape[0]} muestras ({y_test.sum()} defaults, {(1-y_test.mean())*100:.0f}% pagadores)")
print(f"     Proporción target en train: {y_train.mean():.3f}")
print(f"     Proporción target en test:  {y_test.mean():.3f}")

# =============================================================================
# PASO 5: MODELO BASE — UN SOLO ÁRBOL DE DECISIÓN 🌳
# =============================================================================
print("\n" + "=" * 72)
print("📌 PASO 5: Modelo BASE — Árbol de Decisión (nuestro punto de comparación)")
print("=" * 72)

# Entrenamos un árbol de decisión con profundidad limitada para visualizarlo
# (Como hicimos en el proyecto anterior, pero aquí sirve como baseline)
dt_base = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_base.fit(X_train, y_train)

# Predicciones
y_pred_dt = dt_base.predict(X_test)
y_proba_dt = dt_base.predict_proba(X_test)[:, 1]

# Métricas
dt_acc = accuracy_score(y_test, y_pred_dt)
dt_prec = precision_score(y_test, y_pred_dt)
dt_rec = recall_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt)

print(f"\n  🌳 Árbol de Decisión (max_depth=3):")
print(f"     Accuracy:  {dt_acc:.4f}")
print(f"     Precision: {dt_prec:.4f}")
print(f"     Recall:    {dt_rec:.4f}")
print(f"     F1-Score:  {dt_f1:.4f}")

# --- Visualizamos el árbol ---
plt.figure(figsize=(20, 10))
# Usamos los nombres reales de las características para la visualización
plot_tree(dt_base, feature_names=feature_columns, class_names=['Paga 👍', 'Default 👎'],
          filled=True, rounded=True, fontsize=10)
plt.title('Árbol de Decisión (max_depth=3) — Visualización', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, '03_decision_tree.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ docs/03_decision_tree.png — Visualización del árbol guardada")

# Feature importance del árbol
dt_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': dt_base.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  📈 Importancia de características (Árbol de Decisión):")
for _, row in dt_importance.head(5).iterrows():
    print(f"     {row['feature']:20s}: {row['importance']:.4f} ({row['importance']*100:.1f}%)")

# =============================================================================
# PASO 6: RANDOM FOREST — EL BOSQUE 🌲🌲🌲
# =============================================================================
print("\n" + "=" * 72)
print("📌 PASO 6: Random Forest — ¡100 árboles piensan mejor que 1!")
print("=" * 72)

# --- 6a: Random Forest con parámetros por defecto ---
print("\n   🚀 Entrenando Random Forest...")

# Random Forest combina dos ideas clave:
# 1. BAGGING (Bootstrap Aggregating): cada árbol se entrena con una muestra
#    aleatoria diferente de los datos (con reemplazo)
# 2. RANDOM SUBSET OF FEATURES: en cada división, solo se considera un
#    subconjunto aleatorio de características
#
# Esto hace que los árboles sean DIFERENTES entre sí. Al promediar sus
# predicciones, el ruido individual se cancela y queda solo la señal real.

rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)

# Predicciones
y_pred_rf = rf_base.predict(X_test)
y_proba_rf = rf_base.predict_proba(X_test)[:, 1]

# Métricas
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf)
rf_rec = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print(f"\n  🌲 Random Forest (100 árboles, parámetros default):")
print(f"     Accuracy:  {rf_acc:.4f}")
print(f"     Precision: {rf_prec:.4f}")
print(f"     Recall:    {rf_rec:.4f}")
print(f"     F1-Score:  {rf_f1:.4f}")

# --- 6b: Comparación directa ---
print("\n" + "-" * 60)
print("  ⚔️  COMPARACIÓN DIRECTA: Árbol Simple vs Random Forest")
print("-" * 60)
print(f"  {'Métrica':<15s} {'🌳 Árbol':<15s} {'🌲 Random Forest':<15s} {'Mejora':<10s}")
print("-" * 60)

metrics = [
    ('Accuracy', dt_acc, rf_acc),
    ('Precision', dt_prec, rf_prec),
    ('Recall', dt_rec, rf_rec),
    ('F1-Score', dt_f1, rf_f1),
]

for name, dt_val, rf_val in metrics:
    improvement = (rf_val - dt_val) / dt_val * 100
    arrow = '🟢' if improvement > 0 else '🔴'
    print(f"  {name:<15s} {dt_val:<15.4f} {rf_val:<15.4f} {arrow} {improvement:+.2f}%")

print("-" * 60)

# =============================================================================
# PASO 7: FEATURE IMPORTANCE — ¿QUÉ APRENDIÓ EL BOSQUE? 🔍
# =============================================================================
print("\n" + "=" * 72)
print("📌 PASO 7: Importancia de características — ¿qué mira el bosque?")
print("=" * 72)

# --- 7a: Feature importance del Random Forest ---
rf_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_base.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  📊 Top 5 características más importantes (Random Forest):")
for i, (_, row) in enumerate(rf_importance.head(5).iterrows(), 1):
    print(f"     {i}. {row['feature']:20s}: {row['importance']:.4f} ({row['importance']*100:.1f}%)")

print(f"\n  📊 Características menos importantes:")
for _, row in rf_importance.tail(3).iterrows():
    print(f"     • {row['feature']:20s}: {row['importance']:.4f} ({row['importance']*100:.1f}%)")

# --- 7b: Visualización comparativa de feature importance ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Árbol de Decisión
dt_imp_sorted = dt_importance.sort_values('importance')
axes[0].barh(dt_imp_sorted['feature'], dt_imp_sorted['importance'], color='#2ECC71', edgecolor='white')
axes[0].set_title('🌳 Árbol de Decisión', fontweight='bold')
axes[0].set_xlabel('Importancia')
axes[0].set_xlim(0, max(dt_imp_sorted['importance'].max(), rf_importance['importance'].max()) * 1.1)

# Random Forest
rf_imp_sorted = rf_importance.sort_values('importance')
axes[1].barh(rf_imp_sorted['feature'], rf_imp_sorted['importance'], color='#3498DB', edgecolor='white')
axes[1].set_title('🌲 Random Forest (100 árboles)', fontweight='bold')
axes[1].set_xlabel('Importancia')
axes[1].set_xlim(0, max(dt_imp_sorted['importance'].max(), rf_importance['importance'].max()) * 1.1)

fig.suptitle('Comparación de Importancia de Características', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ docs/04_feature_importance.png — Comparación de importancia guardada")

# =============================================================================
# PASO 8: MATRICES DE CONFUSIÓN 🎯
# =============================================================================
print("\n" + "=" * 72)
print("📌 PASO 8: Matrices de confusión — ¿dónde se equivoca cada modelo?")
print("=" * 72)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Árbol de Decisión
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=['Paga 👍', 'Default 👎'],
            yticklabels=['Paga 👍', 'Default 👎'])
axes[0].set_title(f'🌳 Árbol de Decisión\nAccuracy: {dt_acc:.3f}', fontweight='bold')
axes[0].set_ylabel('Real')
axes[0].set_xlabel('Predicho')

# Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Paga 👍', 'Default 👎'],
            yticklabels=['Paga 👍', 'Default 👎'])
axes[1].set_title(f'🌲 Random Forest\nAccuracy: {rf_acc:.3f}', fontweight='bold')
axes[1].set_ylabel('Real')
axes[1].set_xlabel('Predicho')

fig.suptitle('Matrices de Confusión — Comparación', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ docs/05_confusion_matrices.png — Matrices de confusión guardadas")

# Analizamos los errores
print(f"\n  📋 Análisis de errores (Random Forest):")
tn, fp, fn, tp = cm_rf.ravel()
print(f"     ✅ Verdaderos Negativos (pagó correcto): {tn}")
print(f"     ❌ Falsos Positivos (predijo default, pero pagó): {fp}")
print(f"     ❌ Falsos Negativos (predijo pagó, pero fue default): {fn}")
print(f"     ✅ Verdaderos Positivos (default correcto): {tp}")
print(f"     \n     💡 En préstamos: FN (decir que pagará pero no paga) cuesta DINERO al banco.")
print(f"     💡 Es mejor FP (investigar más) que FN (perder el dinero).")

# =============================================================================
# PASO 9: CURVAS ROC Y AUC 📈
# =============================================================================
print("\n" + "=" * 72)
print("📌 PASO 9: Curvas ROC — midiendo la calidad del clasificador")
print("=" * 72)

# Calculamos la curva ROC para ambos modelos
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_dt = auc(fpr_dt, tpr_dt)
auc_rf = auc(fpr_rf, tpr_rf)

fig, ax = plt.subplots(figsize=(10, 8))

# Plot de las curvas ROC
ax.plot(fpr_dt, tpr_dt, '--', color='#2ECC71', linewidth=2,
        label=f'🌳 Árbol Decisión (AUC = {auc_dt:.3f})')
ax.plot(fpr_rf, tpr_rf, '-', color='#3498DB', linewidth=2.5,
        label=f'🌲 Random Forest (AUC = {auc_rf:.3f})')

# Línea diagonal (clasificador aleatorio)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Aleatorio (AUC = 0.5)')

ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12)
ax.set_ylabel('Tasa de Verdaderos Positivos (Recall)', fontsize=12)
ax.set_title('Curvas ROC — Comparación de Modelos', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ docs/06_roc_curves.png — Curvas ROC guardadas")
print(f"\n  📊 AUC (Área Bajo la Curva):")
print(f"     🌳 Árbol de Decisión: {auc_dt:.4f}")
print(f"     🌲 Random Forest:      {auc_rf:.4f}")
print(f"     🟢 Mejora: {(auc_rf - auc_dt) / auc_dt * 100:+.2f}%")

# =============================================================================
# PASO 10: VALIDACIÓN CRUZADA 🧪
# =============================================================================
print("\n" + "=" * 72)
print("📌 PASO 10: Validación Cruzada (5-fold) — ¿qué tan consistentes son?")
print("=" * 72)

# Evaluamos ambos modelos con validación cruzada para ver su consistencia
# La validación cruzada divide los datos en 5 partes, entrena en 4 y prueba en 1,
# repitiendo 5 veces. Así vemos si el modelo es consistente.

cv_scores_dt = cross_val_score(dt_base, X_train, y_train, cv=5, scoring='accuracy')
cv_scores_rf = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                               X_train, y_train, cv=5, scoring='accuracy')

print(f"\n  🌳 Árbol de Decisión — CV (5-fold):")
print(f"     Scores: {cv_scores_dt.round(4)}")
print(f"     Media:  {cv_scores_dt.mean():.4f} ± {cv_scores_dt.std():.4f}")

print(f"\n  🌲 Random Forest — CV (5-fold):")
print(f"     Scores: {cv_scores_rf.round(4)}")
print(f"     Media:  {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

# La desviación estándar baja del RF indica que es más consistente
print(f"\n  📊 Análisis de consistencia:")
print(f"     DT std:  {cv_scores_dt.std():.4f} (mayor = menos consistente)")
print(f"     RF std:  {cv_scores_rf.std():.4f} (menor = más consistente 🔥)")
print(f"     💡 Random Forest es más ROBUSTO porque el promedio reduce la varianza")

# =============================================================================
# PASO 11: AJUSTE DE HIPERPARÁMETROS ⚙️
# =============================================================================
print("\n" + "=" * 72)
print("📌 PASO 11: Ajuste de hiperparámetros — ¿cuántos árboles necesitamos?")
print("=" * 72)

# Probamos diferentes cantidades de árboles para ver cómo mejora el modelo
n_estimators_range = [1, 5, 10, 25, 50, 100, 200, 500]
train_scores = []
test_scores = []

print("\n   Probando diferentes cantidades de árboles...")
for n in n_estimators_range:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, rf_temp.predict(X_train))
    test_acc = accuracy_score(y_test, rf_temp.predict(X_test))
    
    train_scores.append(train_acc)
    test_scores.append(test_acc)
    
    gap = train_acc - test_acc
    marker = '🟢' if gap < 0.05 else '⚠️' if gap < 0.10 else '🔥'
    print(f"     Árboles: {n:3d} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | Gap: {gap:.4f} {marker}")

# --- Visualización: Árboles vs Accuracy ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(n_estimators_range, train_scores, 'o-', color='#2ECC71', linewidth=2, label='Train Accuracy')
ax.plot(n_estimators_range, test_scores, 's-', color='#3498DB', linewidth=2, label='Test Accuracy')
ax.axhline(y=max(test_scores), color='red', linestyle='--', alpha=0.5,
           label=f'Mejor Test: {max(test_scores):.4f}')

ax.set_xlabel('Número de Árboles (n_estimators)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Evolución del Rendimiento vs Número de Árboles', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')  # Escala logarítmica para ver mejor los primeros valores
ax.set_xticks(n_estimators_range)
ax.set_xticklabels([str(x) for x in n_estimators_range])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_tuning_n_estimators.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ docs/07_tuning_n_estimators.png — Gráfico de tuning guardado")

# --- Grid Search rápido sobre max_depth ---
print("\n   💡 Dato importante: con 100 árboles ya llegamos al rendimiento máximo.")
print("   Más árboles no siempre es mejor. 100 es un buen balance calidad/velocidad.")

print("\n   Buscando la profundidad óptima de los árboles...")
param_grid = {'max_depth': [3, 5, 7, 10, 15, None]}
grid_search = GridSearchCV(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\n  🔍 Grid Search Results:")
print(f"     Mejor max_depth: {grid_search.best_params_}")
print(f"     Mejor CV accuracy: {grid_search.best_score_:.4f}")

print(f"\n  Resultados por profundidad:")
for params, mean_score, std_score in zip(
    grid_search.cv_results_['params'],
    grid_search.cv_results_['mean_test_score'],
    grid_search.cv_results_['std_test_score']
):
    print(f"     max_depth={str(params['max_depth']):4s}: {mean_score:.4f} ± {std_score:.4f}")

# =============================================================================
# PASO 12: MODELO FINAL OPTIMIZADO 🏆
# =============================================================================
print("\n" + "=" * 72)
print("📌 PASO 12: Modelo Final — Random Forest con mejores parámetros")
print("=" * 72)

# Entrenamos el modelo final con los mejores parámetros encontrados
best_max_depth = grid_search.best_params_['max_depth']
rf_final = RandomForestClassifier(
    n_estimators=100,
    max_depth=best_max_depth,
    random_state=42,
    n_jobs=-1
)
rf_final.fit(X_train, y_train)

y_pred_final = rf_final.predict(X_test)
y_proba_final = rf_final.predict_proba(X_test)[:, 1]

final_acc = accuracy_score(y_test, y_pred_final)
final_prec = precision_score(y_test, y_pred_final)
final_rec = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)
final_auc = auc(*roc_curve(y_test, y_proba_final)[:2])

print(f"\n  🏆 Random Forest Optimizado:")
print(f"     n_estimators = 100")
print(f"     max_depth    = {best_max_depth}")
print(f"     ---")
print(f"     Accuracy:  {final_acc:.4f}")
print(f"     Precision: {final_prec:.4f}")
print(f"     Recall:    {final_rec:.4f}")
print(f"     F1-Score:  {final_f1:.4f}")
print(f"     AUC-ROC:   {final_auc:.4f}")

# Reporte de clasificación completo
print(f"\n  📋 Reporte de Clasificación Detallado:")
print(classification_report(y_test, y_pred_final,
                            target_names=['Paga 👍', 'Default 👎']))

# =============================================================================
# 🧠 RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 72)
print("  🧠 RESUMEN — Lo que aprendiste hoy")
print("=" * 72)

print("""
  🌲 RANDOM FOREST — El Bosque Mágico

  📌 ¿Qué es Random Forest?
     • Es un ENSAMBLE de muchos árboles de decisión
     • Cada árbol se entrena con datos diferentes (Bootstrap)
     • Cada árbol solo ve un subconjunto de características
     • La predicción final es el PROMEDIO de todos los árboles
       (voto mayoritario en clasificación, promedio en regresión)

  📌 ¿Por qué funciona mejor que un árbol solo?
     • Reduce la VARIANZA: un árbol puede sobreajustarse, pero
       el promedio de 100 árboles reduce ese riesgo
     • Mantiene el SESGO bajo: cada árbol individual es potente
     • Es más ROBUSTO: cambios pequeños en datos no afectan tanto

  📌 Conceptos clave que viste hoy:
     • 🌳 Decision Tree vs 🌲 Random Forest — comparación directa
     • 📊 Feature Importance — qué características importan más
     • 🎯 Confusion Matrix — dónde se equivoca cada modelo
     • 📈 ROC Curve y AUC — calidad del clasificador
     • 🧪 Cross-Validation — consistencia del modelo
     • ⚙️ Hyperparameter Tuning — cómo ajustar n_estimators y max_depth

  📌 ¿Cuándo usar Random Forest?
     ✅ Clasificación y regresión (funciona para ambos)
     ✅ Datos tabulares con muchas características
     ✅ Cuando necesitas feature importance
     ✅ Cuando la precisión es crítica y tienes suficiente datos
     ❌ Dataset muy pequeños (< 100 muestras)
     ❌ Datos de series temporales (mejor usar modelos específicos)
     ❌ Cuando necesitas explicabilidad total (mejor árbol simple)

  📌 Próximo proyecto: Validación Cruzada y Métricas Avanzadas
     • Más técnicas de validación (K-Fold, Stratified, Leave-One-Out)
     • Métricas para clasificación desbalanceada
     • Curvas de aprendizaje
""")

# Guardamos métricas en un CSV para referencia
metrics_summary = pd.DataFrame({
    'Modelo': ['Árbol Decisión (max_depth=3)', 'Random Forest (default)', 'Random Forest (optimizado)'],
    'Accuracy': [dt_acc, rf_acc, final_acc],
    'Precision': [dt_prec, rf_prec, final_prec],
    'Recall': [dt_rec, rf_rec, final_rec],
    'F1': [dt_f1, rf_f1, final_f1],
    'AUC': [auc_dt, auc_rf, final_auc]
})
metrics_summary.to_csv(os.path.join(OUTPUT_DIR, '..', 'data', 'metrics_summary.csv'), index=False)
print("\n  💾 Métricas guardadas en data/metrics_summary.csv")
print(f"\n  📂 Todos los gráficos en: docs/")
print("=" * 72)
