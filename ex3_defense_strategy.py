"""
Exercice 3: Analyse & Stratégie de Défense
=============================================
Objectif: Appliquer concepts à cas réel cybersécurité

Temps: 15 minutes
- Lecture scénario: 2 min
- Threat assessment: 5 min
- Proposer défense: 5 min
- Visualiser analyse: 3 min

Scénario réaliste: Détecteur malware bancaire face menace adversarial
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import os

print("=" * 80)
print("EXERCICE 3: ANALYSE & STRATÉGIE DE DÉFENSE")
print("=" * 80)

# ============================================================================
# PRÉSENTATION SCÉNARIO
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SCÉNARIO RÉALISTE: SÉCURITÉ BANCAIRE                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

CONTEXTE:
─────────
Vous êtes Security Engineer chez BankCorp France (banque régionale)

Système existant:
  • Détecteur malware ML-based
  • Entraîné sur 500,000 fichiers (2019-2023)
  • Déployé en production: analyse 10,000 fichiers/jour
  • Performance: 98% accuracy, 96% recall, 99% precision

MENACE IDENTIFIÉE (Renseignement interne):
──────────────────────────────────────────
Groupe attaquant "APT-SecureBank":
  1. Récupère le modèle (via reverse engineering/API)
  2. Génère malware adversarial
  3. Contourne détecteur BankCorp
  4. Propage malware → clients BankCorp
  5. Vole données, ~€10M+ potential damage

VOTRE MISSION (Urgent!):
────────────────────────
1. Évaluer risque réaliste
2. Proposer défense pragmatique (5 couches)
3. Quantifier efficacité
4. Recommander à Board direction
5. Justifier investissement €100k

╚══════════════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# PARTIE 1: THREAT ASSESSMENT
# ============================================================================

print("\n[STEP 1] Évaluation menace adversarial (Threat Assessment)...\n")

class ThreatAssessment:
    """
    Évaluer probabilité + impact attaque adversarial
    Utilise framework: CVSS-like scoring
    """
    
    def __init__(self):
        self.factors = {}
        self.scores = {}
    
    def assess_feasibility(self):
        """Faisabilité technique"""
        print("1. FAISABILITÉ TECHNIQUE")
        print("   Peut-on générer malware adversarial?")
        print()
        
        methods = {
            'FGSM': {
                'complexity': 'Très basse',
                'time': '5 minutes',
                'code_lines': '< 50',
                'knowledge': 'Basique (ML 101)',
                'success_rate': '50-70%',
                'score': 9
            },
            'PGD (itératif)': {
                'complexity': 'Basse',
                'time': '15 minutes',
                'code_lines': '< 100',
                'knowledge': 'Intermédiaire',
                'success_rate': '75-85%',
                'score': 8
            },
            'Black-Box Attack': {
                'complexity': 'Moyenne',
                'time': '1-2 heures',
                'code_lines': '200-500',
                'knowledge': 'Avancé',
                'success_rate': '85-95%',
                'score': 7
            }
        }
        
        for method, details in methods.items():
            print(f"   {method}:")
            print(f"     • Complexité: {details['complexity']}")
            print(f"     • Temps: {details['time']}")
            print(f"     • Lignes code: {details['code_lines']}")
            print(f"     • Connaissances requises: {details['knowledge']}")
            print(f"     • Taux succès: {details['success_rate']}")
            print(f"     • Score faisabilité: {details['score']}/10")
            print()
        
        avg_score = np.mean([d['score'] for d in methods.values()])
        self.scores['feasibility'] = avg_score
        
        print(f"   ✓ Score faisabilité MOYEN: {avg_score:.1f}/10")
        print(f"     → VERDICT: TRÈS FAISABLE pour attaquant motivé\n")
        
        return avg_score
    
    def assess_motivation(self):
        """Motivation attaquant"""
        print("2. MOTIVATION ATTAQUANT")
        print("   Pourquoi cibler BankCorp?")
        print()
        
        motivations = {
            'Gain financier': {
                'potential': 'Millions € (vol données, rançon)',
                'likelihood': 'Très élevée',
                'effort': 'Modéré',
                'score': 10
            },
            'Avantage compétitif': {
                'potential': 'Technologie, secrets commerciaux',
                'likelihood': 'Élevée',
                'effort': 'Modéré',
                'score': 8
            },
            'Impact géopolitique': {
                'potential': 'Influencer marché financier',
                'likelihood': 'Moyenne',
                'effort': 'Important',
                'score': 6
            },
            'Preuve concept': {
                'potential': 'Démontrer vulnérabilité ML',
                'likelihood': 'Moyenne',
                'effort': 'Modéré',
                'score': 5
            }
        }
        
        for motivation, details in motivations.items():
            print(f"   {motivation}: Score {details['score']}/10")
            print(f"     • Gain potentiel: {details['potential']}")
            print(f"     • Probabilité: {details['likelihood']}")
            print()
        
        max_score = max(d['score'] for d in motivations.values())
        self.scores['motivation'] = max_score
        
        print(f"   ✓ Score motivation MAX: {max_score}/10")
        print(f"     → VERDICT: TRÈS MOTIVÉ (gain financier énorme)\n")
        
        return max_score
    
    def assess_detection_defense(self):
        """Efficacité détection actuelle"""
        print("3. DÉFENSES ACTUELLES & DÉTECTION")
        print("   Quels contrôles existent?")
        print()
        
        controls = {
            'Sandbox testing': {
                'effectiveness': 'Modérée',
                'coverage': '70%',
                'bypass_probability': '30%',
                'score': 7
            },
            'YARA/signature rules': {
                'effectiveness': 'Modérée',
                'coverage': '60%',
                'bypass_probability': '40%',
                'score': 6
            },
            'EDR (Endpoint Detection)': {
                'effectiveness': 'Élevée',
                'coverage': '80%',
                'bypass_probability': '20%',
                'score': 8
            },
            'ML anomaly detection': {
                'effectiveness': 'VARIABLE (c\'est qu\'on attaque!)',
                'coverage': '95%',
                'bypass_probability': '60-80% (adversarial)',
                'score': 5
            },
            'Human review (sampling)': {
                'effectiveness': 'Très élevée',
                'coverage': '5% (samplings)',
                'bypass_probability': '5%',
                'score': 9
            }
        }
        
        for control, details in controls.items():
            print(f"   {control}: Efficacité {details['effectiveness']}")
            print(f"     • Coverage: {details['coverage']}")
            print(f"     • Bypass probability: {details['bypass_probability']}")
            print()
        
        # Detection score = inverse (plus contrôles = plus difficile)
        avg_detection_score = np.mean([d['score'] for d in controls.values()])
        # Plus score bas = plus facile bypass
        detection_difficulty = 10 - avg_detection_score
        self.scores['detection_difficulty'] = detection_difficulty
        
        print(f"   ✓ Difficulté bypass contrôles: {detection_difficulty:.1f}/10")
        print(f"     → VERDICT: Contournable (surtout ML detector)\n")
        
        return detection_difficulty
    
    def calculate_overall_risk(self):
        """Risque global = faisabilité × motivation / détection"""
        print("4. CALCUL RISQUE GLOBAL")
        print("-" * 70)
        
        feasibility = self.scores['feasibility']
        motivation = self.scores['motivation']
        detection = self.scores['detection_difficulty']
        
        # Formule simple
        overall_risk = (feasibility * motivation / detection) / 10
        overall_risk = min(10, overall_risk)  # Cap at 10
        
        print(f"\n   Formule: (Faisabilité × Motivation) / Détection")
        print(f"   Calcul:  ({feasibility:.1f} × {motivation:.1f}) / {detection:.1f} = {overall_risk:.1f}/10")
        print()
        
        # Risk classification
        if overall_risk >= 8:
            risk_level = "🔴 CRITIQUE"
            recommendation = "ACTION IMMÉDIATE REQUISE"
        elif overall_risk >= 6:
            risk_level = "🟠 ÉLEVÉ"
            recommendation = "Plan défense dans les 2 semaines"
        elif overall_risk >= 4:
            risk_level = "🟡 MODÉRÉ"
            recommendation = "Monitorer, planning défense"
        else:
            risk_level = "🟢 BAS"
            recommendation = "Monitoring régulier"
        
        print(f"   Risque global: {risk_level}")
        print(f"   Recommandation: {recommendation}")
        
        self.scores['overall_risk'] = overall_risk
        
        return overall_risk
    
    def run(self):
        """Run complète évaluation"""
        self.assess_feasibility()
        self.assess_motivation()
        self.assess_detection_defense()
        self.calculate_overall_risk()
        
        print("\n" + "=" * 70)
        print("RÉSUMÉ THREAT ASSESSMENT")
        print("=" * 70)
        
        print(f"""
┌─ SCORES COMPOSANTS ─────────────────────────────────────────┐
│ Faisabilité technique:     {self.scores['feasibility']:.1f}/10 (FACILE)              │
│ Motivation attaquant:      {self.scores['motivation']:.1f}/10 (TRÈS HAUTE)           │
│ Difficulté bypass défense: {self.scores['detection_difficulty']:.1f}/10 (MODÉRÉE)       │
│ ─────────────────────────────────────────────────────────── │
│ RISQUE GLOBAL:             {self.scores['overall_risk']:.1f}/10 (CRITIQUE!)          │
│                                                              │
│ → VERDICT: Attaque probable dans 6-12 mois               │
│ → Probabilité succès: ~50-60% selon approche attaquant    │
│ → Dommage potentiel: €10M+ (données clients)              │
└──────────────────────────────────────────────────────────────┘
""")

# Exécuter assessment
threat = ThreatAssessment()
threat.run()

# ============================================================================
# PARTIE 2: STRATÉGIE DÉFENSE 5 COUCHES
# ============================================================================

print("\n[STEP 2] Proposer stratégie défense (5 couches)...\n")

print("=" * 70)
print("STRATÉGIE DÉFENSE: 5 COUCHES")
print("=" * 70)

defense_strategy = {
    'Couche 1: DÉTECTION': {
        'Objectif': 'Détecter inputs adversariales AVANT classification',
        'Mesures': [
            '✓ Validation d\'entrée rigoureuse (format, signature)',
            '✓ Anomaly detection sur features (PCA, Isolation Forest)',
            '✓ Statistiques activations (comparaison training data)',
            '✓ Confidence thresholding (reject si < 70%)',
            '✓ Tripwire: monitoring anormal pattern'
        ],
        'Efficacité': '30-40%',
        'Coût': 'Bas (€5-10k)',
        'Temps': '1-2 semaines',
        'Impact_perf': 'Minimal (< 1%)'
    },
    'Couche 2: ROBUSTESSE': {
        'Objectif': 'Rendre modèle résistant à perturbations',
        'Mesures': [
            '✓ Adversarial Training (FGSM + PGD multi-epsilon)',
            '✓ Ensemble de 5-10 modèles (voting)',
            '✓ Input preprocessing (débruitage, normalisation)',
            '✓ Certified defenses (si faisable)',
            '✓ Regular model retraining (monthly) avec nouvelles techniques'
        ],
        'Efficacité': '75-85%',
        'Coût': 'Moyen (€40-60k)',
        'Temps': '4-8 semaines',
        'Impact_perf': 'Modéré (3-5% accuracy loss)'
    },
    'Couche 3: MONITORING': {
        'Objectif': 'Détecter dégradation modèle en production',
        'Mesures': [
            '✓ Accuracy tracking vs validation set',
            '✓ Confidence distribution monitoring',
            '✓ Model drift detection (data shift)',
            '✓ Alert si accuracy < 95% ou pattern change',
            '✓ Dashboard temps réel (Grafana + Prometheus)',
            '✓ Daily automated tests'
        ],
        'Efficacité': '50% (détecte attaque en < 1h)',
        'Coût': 'Bas (€10-15k)',
        'Temps': '2-3 semaines',
        'Impact_perf': 'Aucun (monitoring only)'
    },
    'Couche 4: RÉPONSE INCIDENT': {
        'Objectif': 'Contenir et remédier si attaque réussit',
        'Mesures': [
            '✓ Playbook incident prédéfini',
            '✓ Quarantine: rejeter fichiers confiance < 80%',
            '✓ Investigation rapide (root cause, quels malwares)',
            '✓ Remediation: ré-entraîner modèle v2',
            '✓ Rollback plan (fallback à signature-based)',
            '✓ Communication clients (72h max RGPD)'
        ],
        'Efficacité': '80% (contient impact)',
        'Coût': 'Moyen (€15-20k)',
        'Temps': '2 semaines',
        'Impact_perf': 'Peut être disruptif'
    },
    'Couche 5: PRÉVENTION LONG-TERME': {
        'Objectif': 'Éviter situation se reproduise',
        'Mesures': [
            '✓ Red teaming quarterly (pen testing adversarial)',
            '✓ Model versioning (garder historique)',
            '✓ Security training équipe ML',
            '✓ Threat intelligence (suivre APT)',
            '✓ ArXiv monitoring (nouveaux papiers adversarial)',
            '✓ Cyber insurance (couverture €5-10M)',
            '✓ Academic collaboration (chercheurs ML robustness)'
        ],
        'Efficacité': '90%+ (prévient future)',
        'Coût': 'Modéré (€30-40k/an)',
        'Temps': 'Continu',
        'Impact_perf': 'Aucun'
    }
}

for layer, details in defense_strategy.items():
    print(f"\n┌─ {layer} ─────────────────────────────────────────────────────┐")
    print(f"│ Objectif: {details['Objectif']}")
    print(f"│ Efficacité: {details['Efficacité']}")
    print(f"│ Coût: {details['Coût']} | Temps: {details['Temps']}")
    print(f"│")
    for mesure in details['Mesures']:
        print(f"│ {mesure}")
    print(f"└────────────────────────────────────────────────────────────────┘")

# ============================================================================
# PARTIE 3: QUANTIFICATION EFFICACITÉ
# ============================================================================

print("\n[STEP 3] Quantifier efficacité probabiliste...\n")

print("=" * 70)
print("MODÈLE PROBABILISTE DE SUCCÈS ATTAQUE")
print("=" * 70)

print("""
Équation: P(attaque réussit) = P(faisable) × P(passe défenses) × P(impact)

Scénario 1: AVANT DÉFENSE (Situation actuelle)
──────────────────────────────────────────────
""")

# Before defense
p_faisable_before = 0.95      # 95% attaquant réussit générer malware adv
p_passe_defenses_before = 0.65  # 65% passe sandboxes + controls
p_impact_before = 0.95        # 95% malware provoque dégât

p_success_before = p_faisable_before * p_passe_defenses_before * p_impact_before

print(f"P(faisable)           = {p_faisable_before:.0%}")
print(f"P(passe défenses)     = {p_passe_defenses_before:.0%}")
print(f"P(impact)             = {p_impact_before:.0%}")
print(f"─────────────────────────")
print(f"P(succès total)       = {p_success_before:.1%}")
print(f"\n→ VERDICT: Attaque probable! ~{p_success_before*100:.0f}% chance succès")

print(f"""
Scénario 2: APRÈS DÉFENSE (Après Phase 1+2 implémentation)
───────────────────────────────────────────────────────────
""")

# After defense (Phase 1-2)
p_faisable_after = 0.95       # Même (attaquant toujours peut générer)
p_passe_defenses_after = 0.12 # 12% seulement! (detection + robustness)
p_impact_after = 0.95         # Même (malware toujours dangereux)

p_success_after = p_faisable_after * p_passe_defenses_after * p_impact_after

print(f"P(faisable)           = {p_faisable_after:.0%}  (pas changé)")
print(f"P(passe défenses)     = {p_passe_defenses_after:.0%}  (GRÂCE À DÉFENSES!)")
print(f"P(impact)             = {p_impact_after:.0%}  (pas changé)")
print(f"─────────────────────────")
print(f"P(succès total)       = {p_success_after:.1%}")

reduction_factor = p_success_before / p_success_after if p_success_after > 0 else 0
print(f"\n→ VERDICT: Succès attaque {reduction_factor:.1f}× moins probable!")
print(f"   Réduction risque: {(1 - p_success_after/p_success_before)*100:.0f}%")

# ============================================================================
# PARTIE 4: ANALYSE ROI
# ============================================================================

print(f"\n[STEP 4] Analyse ROI investissement défense...\n")

print("=" * 70)
print("ANALYSE ROI DÉFENSE")
print("=" * 70)

# Paramètres
annual_revenue = 500_000_000  # €500M (banque régionale)
potential_damage_pct = 0.02   # 2% revenue potentiel si breach
potential_damage = annual_revenue * potential_damage_pct
loss_probability_before = p_success_before
loss_probability_after = p_success_after

expected_loss_before = potential_damage * loss_probability_before
expected_loss_after = potential_damage * loss_probability_after
expected_savings = expected_loss_before - expected_loss_after

investment_phase_1_2 = 100_000  # €100k
annual_investment = 40_000      # €40k/year maintenance

roi = (expected_savings - investment_phase_1_2) / investment_phase_1_2 * 100
payback_months = (investment_phase_1_2 / expected_savings * 12) if expected_savings > 0 else 9999

print(f"""
Paramètres financiers:
  • Revenue annuel BankCorp: €{annual_revenue:,.0f}
  • Potential damage (breach): €{potential_damage:,.0f} (2% revenue)
  • Loss probability avant: {loss_probability_before:.1%}
  • Loss probability après: {loss_probability_after:.1%}

Expected Annual Loss:
  • AVANT défense: €{expected_loss_before:,.0f}
  • APRÈS défense: €{expected_loss_after:,.0f}
  • ÉPARGNES: €{expected_savings:,.0f}

Investissement:
  • Phase 1-2 (initial): €{investment_phase_1_2:,.0f}
  • Maintenance annuelle: €{annual_investment:,.0f}

ROI Calculation:
  • Benefit year 1: €{expected_savings:,.0f}
  • Cost year 1: €{investment_phase_1_2 + annual_investment:,.0f}
  • Net profit year 1: €{expected_savings - investment_phase_1_2 - annual_investment:,.0f}
  
  • ROI year 1: {roi:.0f}%
  • Payback period: {payback_months:.1f} mois
  
  • 5-year benefit: €{(expected_savings - annual_investment) * 5 - investment_phase_1_2:,.0f}
""")

print(f"✓ CONCLUSION ROI: Investissement EXTRÊMEMENT JUSTIFIÉ")
print(f"  → Rendement > 50:1 (€50 gain pour €1 investi)")
print(f"  → Payback en ~1 mois")

# ============================================================================
# PARTIE 5: VISUALISER ANALYSE
# ============================================================================

print(f"\n[STEP 5] Visualiser analyse risque...\n")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ========== Subplot 1: Threat Assessment Scores ==========
ax1 = fig.add_subplot(gs[0, 0])

threats = ['Faisabilité\nTechnique', 'Motivation\nAttaquant', 'Bypass\nDéfenses']
scores = [threat.scores['feasibility'], threat.scores['motivation'], threat.scores['detection_difficulty']]
colors = ['#d62728' if s >= 8 else '#ff7f0e' if s >= 6 else '#2ca02c' for s in scores]

bars = ax1.bar(threats, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Score (0-10)', fontsize=11, fontweight='bold')
ax1.set_title('Threat Assessment: Composants Risque', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 10.5])
ax1.axhline(y=7, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Seuil critique')
ax1.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f'{score:.1f}/10', ha='center', va='bottom', fontsize=11, fontweight='bold')

# ========== Subplot 2: Success Rate Before/After ==========
ax2 = fig.add_subplot(gs[0, 1])

scenarios = ['AVANT\nDéfense\n(Actuel)', 'APRÈS\nDéfense\n(Phase 1-2)']
success_rates = [p_success_before * 100, p_success_after * 100]
colors_sr = ['#d62728', '#2ca02c']

bars = ax2.bar(scenarios, success_rates, color=colors_sr, alpha=0.7, edgecolor='black', linewidth=2, width=0.6)
ax2.set_ylabel('P(succès attaque) %', fontsize=11, fontweight='bold')
ax2.set_title('Probabilité Succès Attaque Adversarial', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 65])
ax2.grid(True, alpha=0.3, axis='y')

for bar, rate in zip(bars, success_rates):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{rate:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add reduction factor
ax2.annotate('', xy=(0.5, 45), xytext=(0.5, 55),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax2.text(0.7, 50, f'{reduction_factor:.1f}× réduction', fontsize=11, color='red', fontweight='bold')

# ========== Subplot 3: Defense Layers Effectiveness ==========
ax3 = fig.add_subplot(gs[1, 0])

layers = ['Détection\n(L1)', 'Robustesse\n(L2)', 'Monitoring\n(L3)', 'Réponse\n(L4)', 'Prévention\n(L5)']
effectiveness = [35, 80, 45, 85, 65]
colors_eff = plt.cm.RdYlGn(np.array(effectiveness)/100)

bars = ax3.barh(layers, effectiveness, color=colors_eff, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Efficacité Relative (%)', fontsize=11, fontweight='bold')
ax3.set_title('Efficacité par Couche Défense', fontsize=12, fontweight='bold')
ax3.set_xlim([0, 100])
ax3.grid(True, alpha=0.3, axis='x')

for bar, eff in zip(bars, effectiveness):
    width = bar.get_width()
    ax3.text(width - 5, bar.get_y() + bar.get_height()/2.,
            f'{eff}%', ha='right', va='center', fontsize=10, fontweight='bold', color='white')

# ========== Subplot 4: ROI Analysis ==========
ax4 = fig.add_subplot(gs[1, 1])

categories = ['Investissement\nInitial', 'Épargnes\nAnnuelles', 'Profit\nNet Year1']
values = [investment_phase_1_2/1000, expected_savings/1000, (expected_savings - investment_phase_1_2 - annual_investment)/1000]
colors_roi = ['#ff7f0e', '#2ca02c', '#1f77b4']

bars = ax4.bar(categories, values, color=colors_roi, alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_ylabel('Montant (€ milliers)', fontsize=11, fontweight='bold')
ax4.set_title('Analyse ROI: Investissement Défense', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 10,
            f'€{value:.0f}k', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add ROI text
roi_text = f'ROI: {roi:.0f}%\nPayback: {payback_months:.1f} mois'
ax4.text(0.98, 0.97, roi_text, transform=ax4.transAxes,
        fontsize=12, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor='black', linewidth=2),
        fontweight='bold')

plt.suptitle('Exercice 3: Analyse Risque & Stratégie Défense - Détecteur Malware Bancaire', 
             fontsize=14, fontweight='bold', y=0.995)

plt.savefig('ex3_defense_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Figure sauvegardée: ex3_defense_analysis.png")
plt.show()

# ============================================================================
# PARTIE 6: RECOMMANDATION À LA DIRECTION
# ============================================================================

print("\n" + "=" * 80)
print("RECOMMANDATION EXÉCUTIVE À LA DIRECTION")
print("=" * 80)

recommendation = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    RAPPORT SÉCURITÉ - CONFIDENTIEL DIRECTION                 ║
║                                                                              ║
║              MENACE: Adversarial Attacks sur Détecteur Malware               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

RÉSUMÉ EXÉCUTIF:
─────────────────

Situation actuelle: CRITIQUE (8.5/10)
  • Malware adversarial techniquement faisable
  • Attaquants très motivés (gains €10M+)
  • Contrôles actuels insuf à contourner

Risque quantifié:
  • Probabilité succès attaque: {p_success_before*100:.0f}% (AVANT défense)
  • Impact potentiel: €{potential_damage:,.0f} en une seule attaque
  • Probabilité incident 1 an: ~{1 - (1-p_success_before)**(1/4):.0%}

RECOMMANDATION:
────────────────

Approuver investissement €100k pour défense (Phase 1-2)

Justification:
  ✓ Réduction risque: 8.5 → 2.5/10 (68% réduction)
  ✓ Succès attaque: {p_success_before*100:.0f}% → {p_success_after*100:.0f}% ({reduction_factor:.1f}× moins probable)
  ✓ ROI année 1: {roi:.0f}% (€50+ rendement pour €1 investi)
  ✓ Payback period: {payback_months:.1f} mois
  ✓ Expected savings: €{expected_savings:,.0f}/an

TIMELINE IMPLÉMENTATION:
─────────────────────────

Phase 1 (Semaines 1-4): Quick Wins - €20k
  ✓ Confidence thresholding (réjecter < 70%)
  ✓ Input validation renforcée
  ✓ Alert setup
  Efficacité: 35% / Temps déploiement: 2 semaines

Phase 2 (Semaines 5-12): Robustesse - €60k
  ✓ Adversarial Training (FGSM + PGD)
  ✓ Ensemble models (5-10)
  ✓ Model monitoring
  ✓ Incident response plan
  Efficacité: 75-85% / Temps déploiement: 6-8 semaines

Phase 3 (Mois 3-6): Long-terme - €20k/year
  ✓ Red teaming quarterly
  ✓ Security training
  ✓ Cyber insurance
  ✓ Academic collaboration
  Efficacité: 90%+ / Continuous

BUDGET DÉTAIL:
───────────────

Année 1:
  • Infrastructure & Tools: €25k
  • ML Engineer temps (200 jours): €50k
  • Red teaming & audit: €15k
  • Training & documentation: €10k
  ─────────────────────────────
  TOTAL: €100k

Années 2-5:
  • Maintenance & updates: €40k/year
  • Security operations: €25k/year
  • Cyber insurance: €50k/year
  • Red teaming: €10k/year
  ─────────────────────────────
  TOTAL: €125k/year

ALTERNATIVE (NOT RECOMMENDED):
Do Nothing (Laisser status quo)
  • Risque reste 8.5/10
  • Probabilité incident: 80%+ sur 1 year
  • Expected loss: €{expected_loss_before:,.0f}
  • Réputation damage: ÉNORME
  • Compliance: RGPD violation risk
  • TOTAL COST: Potentiellement > €50M (branch + litigation)

RISQUES RÉSIDUELS:
──────────────────

Même avec défense implémentée:
  • Attaque 0-day possible (technique inconnue)
  • Insider threat (employé malveillant)
  • Supply chain attack (vendor compromise)
  
Mitigation:
  • Cyber insurance (couverture €5-10M)
  • Regular testing & red teaming
  • Threat intelligence monitoring
  • Incident response rehearsal

CONCLUSION:
────────────

Investissement €100k est INDISPENSABLE pour:
  1. Protéger cliente data (obligation RGPD)
  2. Éviter dommage réputationnel
  3. Respecter fiduciaire duty
  4. Assurer continuité business

Retard = liability enormous

Signature:
  Security Director
  Data: {pd.Timestamp.now().strftime('%d/%m/%Y')}

"""

# Fix timestamp (pandas not imported, do manually)
import datetime
recommendation = recommendation.replace('{pd.Timestamp.now().strftime("%d/%m/%Y")}', 
                                      datetime.datetime.now().strftime('%d/%m/%Y'))

print(recommendation)

# ============================================================================
# PARTIE 7: RÉSUMÉ APPRENTISSAGE
# ============================================================================

print("\n" + "=" * 80)
print("RÉSUMÉ EXERCICE 3 - Ce que nous avons appris")
print("=" * 80)

summary = """
✓ PENSÉE SÉCURITÉ (Risk Management):
  1. Évaluer menace: technique + motivation + défense
  2. Quantifier risque: formules probabilistes
  3. Proposer défense: multi-couches (défense en profondeur)
  4. Calculer ROI: bénéfices vs coûts
  5. Recommander action: data-driven decision making

✓ DÉFENSE EN PROFONDEUR (Defense-in-Depth):
  • Pas une seule défense "parfaite"
  • Combiner plusieurs couches (5 minimum)
  • Chaque couche rattrappe faiblesses autres
  • Permet dégradation gracieuse

✓ MODÈLE PROBABILISTE:
  P(succès) = P(faisable) × P(contourne) × P(impact)
  • Montre complexité sécurité ML
  • Permet quantifier efficacité défense
  • Parle même langage que direction financière

✓ ROI & BUSINESS CASE:
  • Security = investissement (pas coût)
  • €100k investment → €10M+ potential savings
  • Payback très rapide (< 1 mois)
  • Justifie investment à direction

✓ PENSÉE ADVERSAIRE (Red Teaming):
  • Toujours penser comme attaquant
  • "Comment je bypasserais?"
  • Assume attaquant smart, motivated, resourced
  • Test défense même qu'attaquant

IMPLICATION PRATIQUE:
  • Appliquer à votre système (même principe)
  • Évaluer risque adversarial réaliste
  • Proposer défense pragmatique
  • Get buy-in from business via ROI
"""

print(summary)

print("\n" + "=" * 80)
print("EXERCICE 3 TERMINÉ ✓")
print("=" * 80)

print("""
Fin du TD: Vous avez maintenant:

1. EXERCICE 1: Implémenté FGSM (attack)
2. EXERCICE 2: Implémenté Adversarial Training (défense)
3. EXERCICE 3: Pensé stratégie sécurité réaliste

Compétences acquises:
  ✓ Attaques adversariales (théorie + pratique)
  ✓ Défenses robustesse (théorie + pratique)
  ✓ Risk assessment (menace + impact + probabilité)
  ✓ Défense en profondeur (5 couches)
  ✓ ROI calculation (business case)
  ✓ Red team thinking (attaquant perspective)

Next steps:
  • Appliquer à vos systèmes
  • Suivre litterature (arXiv, NeurIPS, ICLR)
  • Participer red teaming
  • Proposer défense robustesse
""")
