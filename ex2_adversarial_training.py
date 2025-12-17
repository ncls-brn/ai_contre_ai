"""
Exercice 2: Adversarial Training
==================================
Objectif: Créer modèles robustes contre adversarial examples

Temps: 20 minutes
- Entraînement: 10 min
- Évaluation: 5 min
- Comparaison: 5 min

Ce script compare un modèle standard vs un modèle entraîné
avec adversarial examples (adversarial training).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

print("=" * 70)
print("EXERCICE 2: Adversarial Training")
print("=" * 70)

# ============================================================================
# PARTIE 1: SETUP DONNÉES (SUBSET POUR VITESSE CPU)
# ============================================================================

print("\n[STEP 1] Charger MNIST (subset pour vitesse CPU)...")

transform = transforms.Compose([transforms.ToTensor()])

# Créer répertoire data si absent
os.makedirs('./data', exist_ok=True)

# Charger complet
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Prendre subset pour vitesse (CPU-only)
train_subset = Subset(mnist_train, range(5000))  # 5000 samples (au lieu de 60000)
test_subset = Subset(mnist_test, range(1000))    # 1000 samples (au lieu de 10000)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

print(f"✓ Train: {len(train_subset)} samples (subset pour vitesse CPU)")
print(f"✓ Test: {len(test_subset)} samples")
print(f"  Note: Subset utilisé pour CPU speed (normally 60k/10k)")

# ============================================================================
# PARTIE 2: DÉFINIR ARCHITECTURE CNN
# ============================================================================

print("\n[STEP 2] Définir architecture CNN...")

class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST
    Même architecture qu'exercice 1
    """
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Créer deux modèles identiques (architecture identique, poids différents)
model_normal = SimpleCNN()
model_robust = SimpleCNN()

print("✓ 2 modèles CNN créés (architecture identique)")
print("  - model_normal: entraîné normalement")
print("  - model_robust: entraîné avec adversarial training")

# ============================================================================
# PARTIE 3: IMPLÉMENTER FGSM (pour adversarial training)
# ============================================================================

print("\n[STEP 3] Implémenter FGSM...")

def fgsm_attack(model, batch, labels, epsilon=0.3):
    """
    Générer adversarial examples via FGSM
    
    Paramètres:
        model: Classifier
        batch: Input batch (batch_size, 1, 28, 28)
        labels: True labels (batch_size,)
        epsilon: Perturbation magnitude
    
    Retour:
        adversarial_batch: x + perturbation, clipped [0,1]
    """
    
    batch_copy = batch.clone().detach()
    batch_copy.requires_grad = True
    
    # Forward pass
    output = model(batch_copy)
    loss = F.cross_entropy(output, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Perturbation
    perturbation = epsilon * torch.sign(batch_copy.grad)
    adversarial_batch = batch_copy + perturbation
    
    return torch.clamp(adversarial_batch, 0, 1).detach()

print("✓ FGSM définie")

# ============================================================================
# PARTIE 4: ENTRAÎNEMENT NORMAL (BASELINE)
# ============================================================================

print("\n[STEP 4] Entraîner modèle NORMAL (clean data only)...\n")

def train_normal(model, train_loader, epochs=3, lr=1e-3):
    """
    Entraînement standard: utiliser que clean data
    
    Paramètres:
        model: Neural network
        train_loader: DataLoader avec training data
        epochs: Nombre passes sur data
        lr: Learning rate
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Normal - Epoch {epoch+1}/{epochs}", leave=True)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Progress bar
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100. * correct / total
            pbar.set_postfix({'loss': f'{avg_loss:.3f}', 'acc': f'{acc:.1f}%'})
    
    return model

model_normal = train_normal(model_normal, train_loader, epochs=3)
print("✓ Modèle normal entraîné\n")

# ============================================================================
# PARTIE 5: ADVERSARIAL TRAINING (ROBUST)
# ============================================================================

print("[STEP 5] Entraîner modèle ROBUSTE (avec adversarial training)...\n")

def train_adversarial(model, train_loader, epochs=3, epsilon=0.3, lr=1e-3):
    """
    Adversarial Training: entraîner sur clean + adversarial examples
    
    Approche:
    - Pour chaque batch:
      1. Générer adversarial examples (FGSM)
      2. Calculer loss sur clean examples
      3. Calculer loss sur adversarial examples
      4. Loss total = 0.5*loss_clean + 0.5*loss_adv
      5. Mise à jour poids
    
    Paramètres:
        model: Neural network
        train_loader: DataLoader
        epochs: Nombre passes
        epsilon: Perturbation magnitude FGSM
        lr: Learning rate
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct_clean = 0
        correct_adv = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Adversarial - Epoch {epoch+1}/{epochs}", leave=True)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            
            # ========== PARTIE CLEAN DATA ==========
            outputs_clean = model(images)
            loss_clean = criterion(outputs_clean, labels)
            
            # ========== GÉNÉRER ADVERSARIAL ==========
            # Générer adversarial examples avec FGSM
            images_adv = fgsm_attack(model, images, labels, epsilon=epsilon)
            
            # ========== PARTIE ADVERSARIAL DATA ==========
            # Prédictions sur adversarial
            outputs_adv = model(images_adv)
            loss_adv = criterion(outputs_adv, labels)
            
            # ========== LOSS COMBINÉ ==========
            # Balancer entre clean et adversarial
            # 50% clean, 50% adversarial
            loss = 0.5 * loss_clean + 0.5 * loss_adv
            
            # ========== BACKWARD PASS ==========
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            _, pred_clean = outputs_clean.max(1)
            correct_clean += pred_clean.eq(labels).sum().item()
            
            _, pred_adv = outputs_adv.max(1)
            correct_adv += pred_adv.eq(labels).sum().item()
            
            total += labels.size(0)
            
            # Progress bar
            avg_loss = total_loss / (batch_idx + 1)
            acc_clean = 100. * correct_clean / total
            acc_adv = 100. * correct_adv / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.3f}',
                'acc_c': f'{acc_clean:.1f}%',
                'acc_a': f'{acc_adv:.1f}%'
            })
    
    return model

model_robust = train_adversarial(model_robust, train_loader, epochs=3, epsilon=0.3)
print("✓ Modèle robust entraîné\n")

# ============================================================================
# PARTIE 6: ÉVALUATION ROBUSTESSE
# ============================================================================

print("[STEP 6] Évaluer robustesse des deux modèles...")
print("(Cela peut prendre 1-2 minutes, merci de patienter)\n")

def evaluate_robustness(model, test_loader, epsilon_list, model_name="Model"):
    """
    Tester robustesse modèle contre différents epsilon
    
    Pour chaque epsilon:
    - Tester accuracy sur clean data
    - Tester accuracy sur adversarial data
    
    Retour:
        dict avec epsilon, clean_acc, adv_acc
    """
    results = {'epsilon': [], 'clean_acc': [], 'adv_acc': []}
    
    model.eval()
    
    for epsilon in epsilon_list:
        correct_clean = 0
        correct_adv = 0
        total = 0
        
        pbar = tqdm(test_loader, desc=f"{model_name} - ε={epsilon:.2f}", leave=False)
        
        for images, labels in pbar:
            
            # Clean accuracy
            with torch.no_grad():
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct_clean += predicted.eq(labels).sum().item()
            
            # Adversarial accuracy
            images_adv = fgsm_attack(model, images, labels, epsilon=epsilon)
            with torch.no_grad():
                outputs = model(images_adv)
                _, predicted = outputs.max(1)
                correct_adv += predicted.eq(labels).sum().item()
            
            total += labels.size(0)
        
        results['epsilon'].append(epsilon)
        results['clean_acc'].append(100. * correct_clean / total)
        results['adv_acc'].append(100. * correct_adv / total)
    
    return results

# Test des deux modèles
epsilon_range = [0.0, 0.1, 0.2, 0.3, 0.4]

print("Évaluation Modèle NORMAL:")
results_normal = evaluate_robustness(model_normal, test_loader, epsilon_range, "Normal")

print("\nÉvaluation Modèle ROBUST:")
results_robust = evaluate_robustness(model_robust, test_loader, epsilon_range, "Robust")

# Afficher tableau comparatif
print("\n" + "=" * 70)
print("RÉSULTATS ROBUSTESSE")
print("=" * 70)

print(f"\n{'ε':>6} | {'Normal Clean':>12} | {'Normal Adv':>11} | {'Robust Clean':>12} | {'Robust Adv':>11}")
print("-" * 70)

for i, epsilon in enumerate(epsilon_range):
    print(f"{epsilon:6.2f} | {results_normal['clean_acc'][i]:11.1f}% | {results_normal['adv_acc'][i]:10.1f}% | {results_robust['clean_acc'][i]:11.1f}% | {results_robust['adv_acc'][i]:10.1f}%")

# ============================================================================
# PARTIE 7: VISUALISER COMPARAISON
# ============================================================================

print("\n[STEP 7] Visualiser comparaison...\n")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# ========== Subplot 1: Adversarial Accuracy Comparison ==========
ax = axes[0]
ax.plot(results_normal['epsilon'], results_normal['adv_acc'], 'ro-', linewidth=2.5, 
        markersize=9, label='Standard Model', markeredgecolor='darkred', markeredgewidth=1.5)
ax.plot(results_robust['epsilon'], results_robust['adv_acc'], 'go-', linewidth=2.5, 
        markersize=9, label='Adversarial Training', markeredgecolor='darkgreen', markeredgewidth=1.5)
ax.set_xlabel('Epsilon (Perturbation Magnitude)', fontsize=12, fontweight='bold')
ax.set_ylabel('Adversarial Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Robustness: Standard vs Adversarial Training', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([0, 105])
ax.set_xlim([-0.05, 0.45])

# Add annotations
for i, epsilon in enumerate(epsilon_range):
    ax.annotate(f'{results_robust["adv_acc"][i]:.0f}%', 
                xy=(epsilon, results_robust['adv_acc'][i]),
                xytext=(0, 5), textcoords='offset points', 
                ha='center', fontsize=9, color='darkgreen', fontweight='bold')

# ========== Subplot 2: Trade-off Accuracy ==========
ax = axes[1]
x = np.arange(len(epsilon_range))
width = 0.35

bars1 = ax.bar(x - width/2, results_robust['clean_acc'], width, 
               label='Clean Accuracy', color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, results_robust['adv_acc'], width, 
               label='Adversarial Accuracy', color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Epsilon (Perturbation Magnitude)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Trade-off: Robustness vs Clean Accuracy', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"{e:.1f}" for e in epsilon_range])
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim([0, 105])

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('ex2_adversarial_training_results.png', dpi=150, bbox_inches='tight')
print("✓ Figure sauvegardée: ex2_adversarial_training_results.png")
plt.show()

# ============================================================================
# PARTIE 8: ANALYSE DÉTAILLÉE DES RÉSULTATS
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSE DÉTAILLÉE DES RÉSULTATS")
print("=" * 70)

# Comparaison à epsilon 0.3
epsilon_idx = 3  # ε=0.3
epsilon_val = 0.3

print(f"\nComparaison détaillée à ε={epsilon_val}:")
print("-" * 70)
print(f"{'Métrique':<35} | {'Normal':>10} | {'Robust':>10} | {'Gain':>10}")
print("-" * 70)

# Clean accuracy
clean_normal = results_normal['clean_acc'][epsilon_idx]
clean_robust = results_robust['clean_acc'][epsilon_idx]
diff_clean = clean_robust - clean_normal
print(f"{'Clean Accuracy':<35} | {clean_normal:>9.1f}% | {clean_robust:>9.1f}% | {diff_clean:>+9.1f}%")

# Adversarial accuracy
adv_normal = results_normal['adv_acc'][epsilon_idx]
adv_robust = results_robust['adv_acc'][epsilon_idx]
diff_adv = adv_robust - adv_normal
improvement_pct = (diff_adv / adv_normal) * 100 if adv_normal > 0 else 0
print(f"{'Adversarial Accuracy':<35} | {adv_normal:>9.1f}% | {adv_robust:>9.1f}% | {diff_adv:>+9.1f}%")
print(f"{'  (% improvement)':<35} | {'':>10} | {'':>10} | {improvement_pct:>+9.0f}%")

# Robustness ratio
robust_normal = (adv_normal / clean_normal) if clean_normal > 0 else 0
robust_robust = (adv_robust / clean_robust) if clean_robust > 0 else 0
diff_ratio = robust_robust - robust_normal
print(f"{'Robustness Ratio (Adv/Clean)':<35} | {robust_normal:>9.1%} | {robust_robust:>9.1%} | {diff_ratio:>+9.1%}")

# Loss of clean accuracy (trade-off)
trade_off = clean_normal - clean_robust
print(f"\nTrade-off Analysis:")
print(f"  Perte clean accuracy: {trade_off:.1f}%")
if trade_off > 0:
    print(f"  → Acceptable: {trade_off:.1f}% accuracy perdue pour {diff_adv:.1f}% robustesse gagnée")
else:
    print(f"  → Bonne nouvelle: Pas de perte clean accuracy!")

print(f"\n✓ CONCLUSION à ε={epsilon_val}:")
if adv_robust > adv_normal:
    print(f"  ✓ Adversarial Training FONCTIONNE!")
    print(f"  ✓ Robustesse améliore de {diff_adv:.1f}% ({improvement_pct:.0f}% relative)")
    print(f"  ✓ Modèle robust peut classifier correctement ~{adv_robust:.0f}% des adversarial examples")
    print(f"  ✗ MAIS modèle normal échoue à ~{100-adv_normal:.0f}%")
else:
    print(f"  ✗ Adversarial Training ne montre pas d'amélioration")

# ============================================================================
# PARTIE 9: INSIGHTS THÉORIQUES
# ============================================================================

print("\n" + "=" * 70)
print("INSIGHTS THÉORIQUES")
print("=" * 70)

print(f"""
Pourquoi Adversarial Training fonctionne:

1. MÉCANISME:
   - Pendant training, modèle voit AUSSI des adversarial examples
   - Apprend à classer correctement même avec perturbations
   - Decision boundaries deviennent plus "lisses"

2. LIMITES:
   - Robustesse spécifique à attaque utilisée (FGSM ici)
   - Si entraîné FGSM, peut ne pas être robust à PGD
   - Perte de performance clean accuracy (ici: {trade_off:.1f}%)

3. EFFICACITÉ:
   - à petit epsilon (< 0.1): très efficace
   - à grand epsilon (> 0.3): réduction rapide efficacité

4. COÛT COMPUTATIONNEL:
   - Adversarial training ≈ 2× plus coûteux que normal training
   - Car génère adversarial PUIS passe forward
   - Valide pour systèmes critiques

5. GÉNÉRALISABILITÉ:
   À TESTER (défi bonus):
   - Modèle entraîné FGSM, robust contre PGD?
   - Transférabilité à autre modèle?
""")

# ============================================================================
# PARTIE 10: RÉSUMÉ
# ============================================================================

print("\n" + "=" * 70)
print("RÉSUMÉ EXERCICE 2")
print("=" * 70)

print(f"""
Ce que nous avons appris:

✓ Adversarial Training est SIMPLE:
  - Générer adversarial examples
  - Entraîner dessus comme normal data
  - Loss = 0.5*loss_clean + 0.5*loss_adv
  - Fonctionne!

✓ Modèles robustes MOINS performants:
  - Clean accuracy: {clean_normal:.1f}% → {clean_robust:.1f}% (perte {trade_off:.1f}%)
  - Mais robustesse énorme: {adv_normal:.1f}% → {adv_robust:.1f}%

✓ Trade-off INÉVITABLE:
  - Robustesse vs Performance
  - Dépend application
  - Critique systems: valide sacrifice

✓ Limitations IMPORTANTES:
  - Robustesse à FGSM seulement
  - Robustesse à PGD? À tester
  - Attaques 0-day? Pas couverte

Exercice suivant: Analyser défense réaliste avec stratégie 5 couches
""")

print("=" * 70)
print("EXERCICE 2 TERMINÉ ✓")
print("=" * 70)
