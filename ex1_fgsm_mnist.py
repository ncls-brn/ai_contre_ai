"""
Exercice 1: FGSM Attack on MNIST
=================================
Objectif: Comprendre et implémenter FGSM

Temps: 25 minutes
- Lecture code: 5 min
- Exécution: 10 min
- Analyse résultats: 10 min

Ce script montre comment générer des adversarial examples
contre un classifier MNIST en utilisant FGSM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

print("=" * 70)
print("EXERCICE 1: FGSM Attack on MNIST")
print("=" * 70)

# ============================================================================
# PARTIE 1: CHARGER DONNÉES MNIST
# ============================================================================

print("\n[STEP 1] Télécharger et charger MNIST...")

# Transform: normaliser images [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Créer répertoire data si absent
os.makedirs('./data', exist_ok=True)

# Télécharger MNIST (automatic download si absent)
mnist_test = datasets.MNIST(
    root='./data',
    train=False,  # Test set
    download=True,
    transform=transform
)

# Data loader (batch_size=1 pour simplicity)
test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)

print(f"✓ MNIST chargé: {len(mnist_test)} images test")
print(f"  Shape: (1, 1, 28, 28)")

# ============================================================================
# PARTIE 2: DÉFINIR ARCHITECTURE CNN SIMPLE
# ============================================================================

print("\n[STEP 2] Définir architecture CNN...")

class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST
    Architecture:
        Conv2d(1, 32, 5) → ReLU → MaxPool2d(2)
        Conv2d(32, 64, 5) → ReLU → MaxPool2d(2)
        Flatten → FC(64*4*4, 128) → ReLU
        FC(128, 10)
    """
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        """Forward pass avec commentaires détaillés"""
        # Conv layer 1
        x = self.conv1(x)                    # (1, 1, 28, 28) → (1, 32, 24, 24)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)               # (1, 32, 24, 24) → (1, 32, 12, 12)
        
        # Conv layer 2
        x = self.conv2(x)                    # (1, 32, 12, 12) → (1, 64, 8, 8)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)               # (1, 64, 8, 8) → (1, 64, 4, 4)
        
        # Flatten
        x = x.view(x.size(0), -1)            # (1, 64*4*4) = (1, 1024)
        
        # FC layers
        x = self.fc1(x)                      # (1, 1024) → (1, 128)
        x = F.relu(x)
        x = self.fc2(x)                      # (1, 128) → (1, 10)
        
        return x

# Instanciate model
model = SimpleCNN()

# Charger ou entraîner poids
weights_path = 'mnist_cnn_weights.pt'

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, weights_only=False))
    print("✓ Poids pré-entraînés chargés")
else:
    print("⚠️  Poids non trouvés - entraînement rapide du modèle (2-3 min)...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Charger training data
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
    
    # Train 1 epoch sur subset
    for epoch in range(1):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Entraînement", leave=False)):
            if batch_idx > 500:  # Only 500 batches for speed
                break
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"  Epoch {epoch+1}: Loss = {total_loss / (batch_idx + 1):.4f}")
    
    torch.save(model.state_dict(), weights_path)
    print("✓ Modèle entraîné et sauvegardé")

model.eval()  # Mode évaluation
print("✓ Modèle prêt")

# ============================================================================
# PARTIE 3: IMPLÉMENTER FGSM
# ============================================================================

print("\n[STEP 3] Implémenter FGSM...\n")

def fgsm_attack(model, image, label, epsilon=0.3):
    """
    Fast Gradient Sign Method (FGSM)
    
    Paramètres:
        model: Neural network classifier
        image: Input image (batch, 1, 28, 28) tensor
        label: True label (batch,) ou scalar tensor
        epsilon: Perturbation magnitude (default 0.3)
    
    Retour:
        adversarial_image: x + perturbation, clipped [0,1]
    
    Mathématique:
        1. Compute gradient: ∇_x L(x, y)
        2. Perturbation: η = ε × sign(∇_x L(x, y))
        3. Adversarial: x' = x + η
        4. Clip: x' = clip(x', 0, 1)
    """
    
    # Step 1: Activer gradient computation pour input
    image_copy = image.clone().detach()
    image_copy.requires_grad = True
    
    # Step 2: Forward pass
    output = model(image_copy)
    
    # Step 3: Calculer loss (CrossEntropy)
    # On veut AUGMENTER la loss (contrairement à training)
    if isinstance(label, int):
        label = torch.tensor([label])
    
    loss = F.cross_entropy(output, label)
    
    # Step 4: Calculer gradient par rapport à x (backprop)
    model.zero_grad()  # Clear previous gradients
    loss.backward()    # Compute ∇_x L(x, y)
    
    # Step 5: Perturbation = sign(gradient)
    perturbation = epsilon * torch.sign(image_copy.grad)
    
    # Step 6: Adversarial example
    adversarial_image = image_copy + perturbation
    
    # Step 7: Clip values [0, 1]
    adversarial_image = torch.clamp(adversarial_image, 0, 1)
    
    return adversarial_image.detach()

print("✓ FGSM définie")

# ============================================================================
# PARTIE 4: TESTER FGSM AVEC DIFFÉRENTS EPSILON
# ============================================================================

print("[STEP 4] Générer adversarial examples...\n")

# Prendre première image du test set
image, label = next(iter(test_loader))
label_int = label.item()
label = torch.tensor([label_int])

print(f"Image originale:")
print(f"  - True label: {label_int} (classe correcte)")
print()

# Test avec différents epsilon
epsilons = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
results = []

print(f"{'ε':>6} | {'Pred':>6} | {'Conf':>7} | {'Succès?':>8} | Perturbation max")
print("-" * 60)

for epsilon in epsilons:
    
    # Générer adversarial example
    if epsilon == 0:
        adversarial_image = image
        perturbation_max = 0.0
    else:
        adversarial_image = fgsm_attack(model, image, label, epsilon=epsilon)
        
        # Calculer perturbation magnitude
        perturbation_max = (adversarial_image - image).abs().max().item()
    
    # Test modèle
    with torch.no_grad():
        output = model(adversarial_image)
        
        # Prédiction
        prediction = output.argmax(dim=1).item()
        
        # Confiance
        confidence = F.softmax(output, dim=1).max().item()
        
        # Succès attaque?
        attack_success = "OUI" if prediction != label_int else "NON"
    
    results.append({
        'epsilon': epsilon,
        'prediction': prediction,
        'confidence': confidence,
        'success': attack_success == "OUI",
        'perturbation_max': perturbation_max
    })
    
    print(f"{epsilon:6.2f} | {prediction:6d} | {confidence:7.1%} | {attack_success:>8} | {perturbation_max:12.4f}")

print()

# ============================================================================
# PARTIE 5: VISUALISER RÉSULTATS
# ============================================================================

print("[STEP 5] Visualiser adversarial examples...\n")

# Créer figure avec subplots (6 colonnes = 6 epsilon)
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
fig.suptitle(f'FGSM Adversarial Examples (True label: {label_int})', 
             fontsize=16, fontweight='bold', y=0.98)

for idx, epsilon in enumerate(epsilons):
    
    # Générer adversarial image
    if epsilon == 0:
        adv_image = image
    else:
        adv_image = fgsm_attack(model, image, label, epsilon=epsilon)
    
    # Row 1: Adversarial image
    ax = axes[0, idx]
    img_array = adv_image.squeeze().numpy()
    ax.imshow(img_array, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'ε={epsilon:.2f}', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Row 2: Perturbation visualization
    ax = axes[1, idx]
    if epsilon == 0:
        perturbation = torch.zeros_like(image)
    else:
        perturbation = (adv_image - image).abs()
    
    pert_array = perturbation.squeeze().numpy()
    im = ax.imshow(pert_array, cmap='hot')
    max_pert = pert_array.max()
    ax.set_title(f'Perturbation\n(max={max_pert:.3f})', fontsize=10)
    ax.axis('off')
    
    # Row 3: Statistics
    ax = axes[2, idx]
    ax.axis('off')
    
    # Get predictions
    with torch.no_grad():
        output = model(adv_image)
        pred = output.argmax(dim=1).item()
        conf = F.softmax(output, dim=1).max().item()
        success = "✓" if pred != label_int else "✗"
    
    # Texte
    text_str = f"Pred: {pred}\nConf: {conf:.1%}\nSuccess: {success}"
    
    bbox_color = 'lightgreen' if pred != label_int else 'lightcoral'
    ax.text(0.5, 0.5, text_str, ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.7, edgecolor='black', linewidth=1.5),
            fontweight='bold')

plt.tight_layout()
plt.savefig('ex1_fgsm_results.png', dpi=150, bbox_inches='tight')
print("✓ Figure sauvegardée: ex1_fgsm_results.png")
plt.show()

# ============================================================================
# PARTIE 6: ANALYSE ET QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSE DES RÉSULTATS")
print("=" * 70)

# Question 1: Threshold d'attaque
successful_attacks = [r for r in results if r['success']]
first_success = successful_attacks[0] if successful_attacks else None

if first_success:
    print(f"\n✓ Question 1: À quel epsilon l'attaque réussit?")
    print(f"  → Premier succès: ε = {first_success['epsilon']:.3f}")
else:
    print(f"\n✗ Question 1: Pas d'attaque réussie détectée (augmenter epsilon?)")

# Question 2: Imperceptibilité
print(f"\n✓ Question 2: Perturbation imperceptible à l'oeil nu?")
if first_success and first_success['perturbation_max'] < 0.15:
    print(f"  → OUI! À ε={first_success['epsilon']}: perturbation = {first_success['perturbation_max']:.4f}")
    print(f"     C'est imperceptible mais suffit à tromper le modèle")
else:
    print(f"  → Perturbations visibles à ces epsilon")

# Question 3: Confiance modèle
success_confidence = [r['confidence'] for r in results if r['success']]
if success_confidence:
    avg_conf = np.mean(success_confidence)
    max_conf = max(success_confidence)
    print(f"\n✓ Question 3: Confiance modèle quand il se trompe?")
    print(f"  → Confiance moyenne attaques réussies: {avg_conf:.1%}")
    print(f"  → Confiance max: {max_conf:.1%}")
    print(f"  → INSIGHT: Modèle reste CONFIANT même quand il se trompe!")
    print(f"     → Impossible de détecter par simple thresholding confiance")
else:
    print(f"\n✗ Pas suffisamment d'exemples adverses pour calculer")

# ============================================================================
# PARTIE 7: RÉSUMÉ
# ============================================================================

print("\n" + "=" * 70)
print("RÉSUMÉ EXERCICE 1")
print("=" * 70)

print(f"""
Ce que nous avons appris:

1. FGSM est SIMPLE:
   - Une ligne mathématique: x' = x + ε × sign(∇L)
   - Très rapide (une seule forward-backward pass)
   - Efficace: ~50-70% taux succès

2. Perturbations IMPERCEPTIBLES:
   - Très petites magnitudes suffisent (ε < 0.15)
   - Modèle classifie complètement différent
   - Humains ne remarquent rien

3. Confiance TROMPEUSE:
   - Modèle reste CONFIANT en prédictions adversariales
   - Simple confiance threshold N'EST PAS défense
   - Besoin de robustesse réelle

4. Implications sécurité:
   ✓ Malware peut ajouter petits changements
   ✓ Passer détecteurs ML
   ✓ Mais reste fonctionnel
   ✓ Problème réel et urgent

Exercice suivant: Comment défendre contre ces attaques?
""")

print("=" * 70)
print("EXERCICE 1 TERMINÉ ✓")
print("=" * 70)
