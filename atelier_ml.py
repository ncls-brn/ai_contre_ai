# NOM_PRENOM_atelier_ml.py

"""
Atelier pratique ML - Détection d'intrusion réseau
Cybersécurité M2

Analyse des résultats attendus (exemple type à adapter) :
- Précision: ~0.95 - 1.00  Peu de fausses alertes, les prédictions "intrusion" sont fiables
- Rappel: ~0.70 - 0.75    Une majorité des intrusions est détectée, mais certaines passent encore
- F1-Score: ~0.80 - 0.85  Bon compromis entre Précision et Rappel
- AUC-ROC: ~0.80 - 0.85   Modèle correctement discriminant

Recommandations possibles :
1) Tester d'autres algorithmes (SVM, Logistic Regression, Gradient Boosting)
2) Travailler sur le déséquilibre de classes (SMOTE, seuil de décision, class_weight)
3) Ajouter des features plus réalistes (fenêtres temporelles, fréquence des connexions, adresses IP, ports)
"""

# ==========================
# Partie 1 - Imports et données
# ==========================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

# Pour reproductibilité
np.random.seed(42)

n_samples = 1000

# Génération de données simulées de trafic réseau
data = {
    "packet_size": np.random.normal(500, 200, n_samples),
    "duration": np.random.exponential(2, n_samples),
    "src_bytes": np.random.lognormal(8, 2, n_samples),
    "dst_bytes": np.random.lognormal(7, 2, n_samples),
    "num_failed_logins": np.random.poisson(0.1, n_samples),
    # Encodage simple du protocole 0 = TCP, 1 = UDP, 2 = ICMP (par exemple)
    "protocol_type": np.random.choice([0, 1, 2], n_samples),
}

df_network = pd.DataFrame(data)

# Nettoyage minimal (éviter valeurs négatives pour packet_size)
df_network["packet_size"] = df_network["packet_size"].clip(lower=0)

# Création des labels d'intrusion basés sur des seuils simples
intrusion_mask = (
    (df_network["packet_size"] > 800)
    | (df_network["duration"] > 5)
    | (df_network["num_failed_logins"] > 2)
)

df_network["is_intrusion"] = intrusion_mask.astype(int)

# ==========================
# Partie 2 - Exploration des données (EDA)
# ==========================

print("Distribution des classes (0 = normal, 1 = intrusion) :")
print(df_network["is_intrusion"].value_counts())
print()

print("Proportion des classes :")
print(df_network["is_intrusion"].value_counts(normalize=True))
print()

print("Statistiques descriptives par classe :")
print(df_network.groupby("is_intrusion").describe())
print()

# Visualisation distribution des classes
plt.figure(figsize=(4, 3))
sns.countplot(x="is_intrusion", data=df_network)
plt.title("Distribution des classes")
plt.xlabel("is_intrusion")
plt.ylabel("Nombre d'échantillons")
plt.tight_layout()
plt.show()

# Histogrammes des principales features par classe
features_to_plot = ["packet_size", "duration", "src_bytes", "dst_bytes", "num_failed_logins"]

for feat in features_to_plot:
    plt.figure(figsize=(5, 3))
    sns.histplot(
        data=df_network,
        x=feat,
        hue="is_intrusion",
        kde=True,
        stat="density",
        common_norm=False
    )
    plt.title(f"Distribution de {feat} par classe")
    plt.tight_layout()
    plt.show()

# ==========================
# Partie 3 - Pipeline ML de base (Random Forest)
# ==========================

# Séparation features / target
X = df_network.drop("is_intrusion", axis=1)
y = df_network["is_intrusion"]

# Division train/test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"  # gestion simple du déséquilibre
)

# Entraînement
rf_model.fit(X_train_scaled, y_train)

# Prédictions
y_pred = rf_model.predict(X_test_scaled)
# Probabilité de la classe positive (intrusion)
if hasattr(rf_model, "predict_proba"):
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
else:
    # fallback rare pour modèles sans predict_proba
    y_scores = rf_model.decision_function(X_test_scaled)
    # min-max pour pseudo proba
    y_pred_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

# ==========================
# Partie 4 - Évaluation
# ==========================

print("Matrice de confusion Random Forest :")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

print("Rapport de classification Random Forest :")
print(classification_report(y_test, y_pred, digits=3))

auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Random Forest : {auc_score:.4f}")

# Heatmap matrice de confusion
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion - Random Forest")
plt.xlabel("Prédiction")
plt.ylabel("Réel")
plt.tight_layout()
plt.show()

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(4, 3))
plt.plot(fpr, tpr, label=f"RF (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC - Random Forest")
plt.legend()
plt.tight_layout()
plt.show()

# Importance des features
importances = rf_model.feature_importances_
feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("Importance des features (Random Forest) :")
print(feature_importance)
print()

plt.figure(figsize=(6, 3))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title("Importance des features - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# ==========================
# Partie 5 - Challenge 1 : GridSearchCV sur RandomForest
# ==========================

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ),
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print("Meilleurs hyperparamètres Random Forest :")
print(grid_search.best_params_)
print(f"Meilleur score F1 (cross-val) : {grid_search.best_score_:.4f}")
print()

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_scaled)
y_pred_best_proba = best_rf.predict_proba(X_test_scaled)[:, 1]
auc_best = roc_auc_score(y_test, y_pred_best_proba)

print("Rapport de classification RF optimisé :")
print(classification_report(y_test, y_pred_best, digits=3))
print(f"AUC-ROC RF optimisé : {auc_best:.4f}")
print()

# ==========================
# Partie 6 - Challenge 2 : Comparaison avec SVM et Logistic Regression
# ==========================

# Logistic Regression
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_lr_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
auc_lr = roc_auc_score(y_test, y_pred_lr_proba)

print("Logistic Regression :")
print(classification_report(y_test, y_pred_lr, digits=3))
print(f"AUC-ROC LR : {auc_lr:.4f}")
print()

# SVM avec probas
svm_clf = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=42
)
svm_clf.fit(X_train_scaled, y_train)
y_pred_svm = svm_clf.predict(X_test_scaled)
y_pred_svm_proba = svm_clf.predict_proba(X_test_scaled)[:, 1]
auc_svm = roc_auc_score(y_test, y_pred_svm_proba)

print("SVM RBF :")
print(classification_report(y_test, y_pred_svm, digits=3))
print(f"AUC-ROC SVM : {auc_svm:.4f}")
print()

# Comparaison visuelle AUC
models_auc = pd.Series(
    {
        "RandomForest (base)": auc_score,
        "RandomForest (best)": auc_best,
        "LogReg": auc_lr,
        "SVM": auc_svm,
    }
)

plt.figure(figsize=(6, 3))
sns.barplot(x=models_auc.index, y=models_auc.values)
plt.xticks(rotation=15)
plt.ylabel("AUC-ROC")
plt.title("Comparaison des modèles (AUC-ROC)")
plt.tight_layout()
plt.show()

# ==========================
# Partie 7 - Challenge 3 : Gestion du déséquilibre (seuil de décision)
# ==========================

# Exemple avec RF optimisé
thresholds_to_test = [0.3, 0.5, 0.7]
print("Impact du seuil de décision sur RF optimisé :")
for th in thresholds_to_test:
    y_pred_th = (y_pred_best_proba >= th).astype(int)
    print(f"\nSeuil = {th}")
    print(confusion_matrix(y_test, y_pred_th))
    print(classification_report(y_test, y_pred_th, digits=3))

print("\nScript terminé.")
