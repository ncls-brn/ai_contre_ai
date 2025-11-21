# Exemples Pédagogiques Complets: Attaques Boostées par IA
**Master 2 Cybersécurité - Formation Éducative Uniquement**

---

## ⚠️ AVERTISSEMENT LÉGAL

### Utilisation Autorisée
```
✅ Compréhension des menaces émergentes
✅ Environnement de formation isolé (air-gapped)
✅ Simulation contrôlée en laboratoire
✅ Sensibilisation équipes cybersécurité
✅ Préparation défenses organisationnelles
```

### Utilisation Interdite
```
❌ Attaques réelles sur systèmes
❌ Cibles non-autorisées
❌ Distribution de code malveillant
❌ Intentions criminelles
❌ Extorsion/fraude/espionnage
```

### Engagement Étudiant
```
Je reconnais que ces exemples sont à fins ÉDUCATIVES uniquement.
Je m'engage à ne pas utiliser ces techniques hors du cadre pédagogique.
Je comprends les implications légales (Art. 323-1 à 323-7, Code Pénal FR).

Signature: ____________  Date: __________
```

---

## 📌 EXEMPLE 1: PHISHING BOOSTÉ PAR IA GÉNÉRATIVE

### Contexte de Menace
- **Acteur**: Attaquant sans expertise rédactionnelle
- **Objectif**: Vol de credentials via email spear-phishing
- **Technique**: LLM + Base CTI + Personalisation
- **Efficacité**: +300% taux réponse vs phishing générique

### Étape 1: Reconnaissance (OSINT)

```python
# reconnaissance.py - PÉDAGOGIQUE UNIQUEMENT
"""
Simulation d'OSINT sur cible consentante (formation)
"""

import requests
import json
from datetime import datetime

class OSINTGatherer:
    """Collecte publique d'informations"""
    
    def __init__(self, target_email="student@ecole.edu"):
        self.target = target_email
        self.domain = target_email.split("@")[1]
        self.collected_data = {}
    
    def gather_info(self):
        """Collecter infos publiques (simulation)"""
        
        # Simulation: Infos disponibles publiquement
        self.collected_data = {
            "email": self.target,
            "domain": self.domain,
            "likely_employer": "École d'Ingénieur XYZ",
            "typical_role": "Étudiant Cybersécurité",
            "social_media_mentions": [
                {"platform": "LinkedIn", "info": "Étude cybersécurité"},
                {"platform": "GitHub", "info": "Intérêt Python/Security"}
            ],
            "public_records": {
                "company_info": "Établissement public",
                "industry": "Éducation",
                "employee_count": "1000+"
            },
            "recent_news": [
                "Nouvelle formation cybersécurité lancée",
                "Partenariat avec entreprise de sécurité"
            ]
        }
        
        return self.collected_data
    
    def build_profile(self):
        """Construire profil attaquant"""
        
        profile = {
            "target": self.target,
            "context": {
                "likely_interests": ["Cybersécurité", "ML", "Défense"],
                "professional_context": "Étudiant/Chercheur",
                "email_domain": self.domain,
                "organization_type": "Académique"
            },
            "vulnerability_vectors": [
                "Curiosité académique",
                "Intérêt nouveaux outils",
                "Confiance pairs académiques",
                "Urgence deadlines"
            ]
        }
        
        return profile

# Utilisation (FORMATION UNIQUEMENT)
osint = OSINTGatherer()
info = osint.gather_info()
profile = osint.build_profile()

print("[OSINT] Données collectées (publiques):")
print(json.dumps(info, indent=2, ensure_ascii=False))
print("\n[PROFIL] Vulnérabilités identifiées:")
print(json.dumps(profile["vulnerability_vectors"], indent=2, ensure_ascii=False))
```

### Étape 2: Génération Email via LLM

```python
# phishing_generation.py - PÉDAGOGIQUE UNIQUEMENT
"""
Génération email phishing personnalisé via LLM
NOTES: Simulation locale, pas d'envoi réel
"""

import ollama
import json

class PhishingEmailGenerator:
    """Génère emails phishing personnalisés (simulation éducative)"""
    
    def __init__(self, llm_model="mistral"):
        self.model = llm_model
        self.generated_emails = []
    
    def generate_pretext(self, profile):
        """Générer prétexte d'attaque convaincant"""
        
        prompt = f"""
Tu es un chercheur en sécurité qui analyse les prétextes d'attaque.
Génère UN prétexte réaliste (simulation éducative) pour une cible:
- Domaine: {profile['context']['email_domain']}
- Rôle: {profile['context']['professional_context']}
- Intérêts: {', '.join(profile['context']['likely_interests'])}

Prétexte doit être:
1. Crédible dans contexte académique
2. Créer urgence/curiosité
3. Sembler d'autorité légitime
4. Basé sur actualités récentes

Format JSON: {{"pretext": "description", "urgency_factor": "haute/moyenne/basse"}}
"""
        
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=False
        )
        
        try:
            pretext = json.loads(response['response'])
            return pretext
        except:
            return {"pretext": "Simulation - pretext generation failed", "urgency_factor": "moyenne"}
    
    def generate_email_body(self, pretext, profile, variant="technical"):
        """Générer corps email personnalisé"""
        
        prompt = f"""
Génère un email phishing RÉALISTE (exercice cybersécurité) basé sur:

Prétexte: {pretext['pretext']}
Cible: {profile['target']}
Variante: {variant} (technical/urgent/authority)

Email doit:
1. Sembler d'une source crédible (ex: admin académique, collègue)
2. Inclure détails personnalisés du profil
3. Créer urgence/curiosité
4. Inclure call-to-action suspecte (lien, formulaire)
5. Être réaliste mais clairement test pédagogique

IMPORTANT: Cet email est pour ANALYSE en environnement formation.
Jamais envoyer.

Format:
Subject: ...
From: ...
Body:
...

Incluez où serait le payload (URL malveillante, formulaire)
"""
        
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            options={'temperature': 0.7}
        )
        
        return response['response']
    
    def generate_variants(self, profile, num_variants=3):
        """Générer variantes d'emails (A/B testing)"""
        
        variants = []
        for i, variant_type in enumerate(["technical", "urgent", "authority"][:num_variants]):
            pretext = self.generate_pretext(profile)
            email = self.generate_email_body(pretext, profile, variant_type)
            
            variants.append({
                "variant": i + 1,
                "type": variant_type,
                "pretext": pretext,
                "email_body": email
            })
        
        return variants

# Utilisation
generator = PhishingEmailGenerator()
profile = osint.build_profile()
pretext = generator.generate_pretext(profile)

print("[PHISHING] Prétexte généré:")
print(json.dumps(pretext, indent=2, ensure_ascii=False))

print("\n[PHISHING] Exemple email (Variante 1):")
email_body = generator.generate_email_body(pretext, profile, "technical")
print(email_body[:500] + "...")

print("\n[NOTICE] Cet email est pour ANALYSE PÉDAGOGIQUE uniquement.")
print("[NOTICE] Ne jamais envoyer à cibles réelles.")
```

### Étape 3: Infrastructure de Capture

```python
# landing_page_simulation.py - PÉDAGOGIQUE UNIQUEMENT
"""
Simulation page landing phishing
IMPORTANT: Jamais déployer en réel - Formation UNIQUEMENT
"""

html_landing_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Mise à jour Sécurité - Authentification Required</title>
    <style>
        body { font-family: Arial; background: #f0f0f0; }
        .container { max-width: 500px; margin: 50px auto; 
                    background: white; padding: 30px; border-radius: 5px; }
        .logo { text-align: center; margin-bottom: 20px; }
        .warning { background: #fff3cd; padding: 10px; border-radius: 3px; }
        input { width: 100%; padding: 10px; margin: 10px 0; box-sizing: border-box; }
        button { width: 100%; padding: 10px; background: #0066cc; color: white; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">
            <h2>Vérification de Compte</h2>
        </div>
        
        <div class="warning">
            <strong>⚠️ Alerte Sécurité</strong><br>
            Votre compte nécessite une vérification d'identité immédiate.
            Veuillez vous réauthentifier.
        </div>
        
        <form id="login-form">
            <label>Email institutionnel:</label>
            <input type="email" name="email" placeholder="prenom.nom@ecole.edu" required>
            
            <label>Mot de passe:</label>
            <input type="password" name="password" placeholder="Votre mot de passe" required>
            
            <label>Code MFA (si activé):</label>
            <input type="text" name="mfa" placeholder="Optionnel">
            
            <button type="submit">Vérifier l'Accès</button>
        </form>
        
        <small style="color: #666; margin-top: 15px;">
            Cette page est une SIMULATION pédagogique.<br>
            Ne jamais entrer vraies identifiants dans des pages non-vérifiées.
        </small>
    </div>
    
    <script>
        // Simulation: Enregistrer données (jamais en réel)
        document.getElementById('login-form').onsubmit = function(e) {
            e.preventDefault();
            console.log("[PÉDAGOGIQUE] Tentative login capturée");
            console.log("Email:", document.querySelector('[name=email]').value);
            console.log("MDP: ***");
            alert("Simulation terminée. En attaque réelle, credentials seraient volés.");
        }
    </script>
</body>
</html>

<!-- ANALYSE:
Indices phishing:
1. ⚠️ Alerte urgente (psych pressure)
2. 🎭 Demande réauthentification (prétexte)
3. 🔗 URL différente de domaine légitime (à analyser)
4. 📱 Demande MFA (paraît moderne/légitime)
5. 📧 Branding imité de site officiel

Défenses contre ce phishing:
- Vérifier vraie URL dans navigateur
- Hover sur liens (voir cible réelle)
- Never entrer credentials sous pression
- Check sender réel du mail
- MFA réel ne demande jamais password
-->
"""

print("[LANDING PAGE] Simulation HTML phishing")
print("Code HTML (fragmenté pour sécurité):")
print(html_landing_page[:800] + "...")
```

### Analyse & Défense

```python
# defense_analysis.py
"""
Analyse des vecteurs d'attaque et défenses
"""

analysis = {
    "attack_vector": {
        "phase_1": "OSINT (reconnaissance publique)",
        "phase_2": "Personalization (LLM génère pretext)",
        "phase_3": "Email craft (hyper-réaliste)",
        "phase_4": "Social engineering (urgence + autorité)",
        "phase_5": "Capture credentials (landing page fausse)"
    },
    
    "efficacité_boosts": {
        "sans_IA": {
            "temps_préparation": "4 heures (manual)",
            "variations_emails": "3-5 templates génériques",
            "taux_réponse": "2-5%",
            "adaptation": "Pas (static)"
        },
        "avec_IA": {
            "temps_préparation": "15 minutes (LLM)",
            "variations_emails": "100+ variants (automated)",
            "taux_réponse": "15-35% (5-7× meilleur)",
            "adaptation": "Oui (dynamic, selon feedback)"
        }
    },
    
    "défenses_détection": [
        {
            "niveau": "1. Technique",
            "mesures": [
                "SPF/DKIM/DMARC validation",
                "ML email filtering (Proofpoint, Mimecast)",
                "URL rewriting + sandboxing",
                "MFA obligatoire (évite credential theft)"
            ]
        },
        {
            "niveau": "2. Organisationnel",
            "mesures": [
                "Formation phishing awareness (régulière)",
                "Simulation phishing interne (realistic)",
                "Zero-trust network (isolation)",
                "Metrics: % utilisateurs cliquant/reporting"
            ]
        },
        {
            "niveau": "3. Comportementale",
            "mesures": [
                "Verifier sender address (not display name)",
                "Hover URLs (voir cible avant cliquer)",
                "Vérifier urgence (attackers utilisent pression)",
                "Demander confirmation hors-band (appel)",
                "Never re-auth après unplanned email"
            ]
        }
    ]
}

print("[DÉFENSE] Comparatif Phishing Classique vs IA-Enhanced:")
print(json.dumps(analysis['efficacité_boosts'], indent=2, ensure_ascii=False))

print("\n[DÉFENSE] Mesures Recommandées:")
for defense_level in analysis['défenses_détection']:
    print(f"\n{defense_level['niveau']}:")
    for measure in defense_level['mesures']:
        print(f"  - {measure}")
```

---

## 📌 EXEMPLE 2: MALWARE POLYMORPHE GÉNÉRÉ PAR IA

### Concept Menace

```python
# malware_polymorphic.py - PÉDAGOGIQUE
"""
Comprendre comment IA génère variantes malware
NOTA: Aucun vrai code malveillant - Structures uniquement
"""

class PolymorphicMalwareAnalysis:
    """
    Analysys pédagogique du polymorphisme généré par IA
    """
    
    def __init__(self):
        self.original_signature = """
        // Pseudo-code original (inoffensif)
        function exfiltrate_data() {
            connect_to_c2("192.168.1.1")
            send_files("/documents/*")
            delete_logs()
        }
        """
        
        self.variations = []
    
    def generate_variations(self):
        """
        Montrer comment IA génère variations
        chacune avec hash différent
        """
        
        variations = [
            {
                "variant": 1,
                "technique": "Code reordering",
                "pseudo_code": "send_files() → delete_logs() → connect_to_c2()",
                "md5_hash": "a1b2c3d4e5f6...",
                "détection": "Signaure classique inefficace"
            },
            {
                "variant": 2,
                "technique": "Variable renaming",
                "pseudo_code": "exfil_data → data_leak, c2 → server",
                "md5_hash": "f6e5d4c3b2a1...",
                "détection": "Requier semantic analysis"
            },
            {
                "variant": 3,
                "technique": "Dead code injection",
                "pseudo_code": "Ajouter loops inutiles, calculs fake",
                "md5_hash": "9z8y7x6w5v4u...",
                "détection": "Polymorphism engine necessary"
            },
            {
                "variant": 4,
                "technique": "API call obfuscation",
                "pseudo_code": "WriteFile() → WriteFileEx() → API table resolve",
                "md5_hash": "3c4d5e6f7g8h...",
                "détection": "Requires behavioral analysis"
            },
            {
                "variant": 5,
                "technique": "Encryption + Dynamic decode",
                "pseudo_code": "XOR key avec timestamp, auto-decode at runtime",
                "md5_hash": "5e6f7g8h9i0j...",
                "détection": "Sandbox + memory analysis"
            }
        ]
        
        return variations
    
    def gan_generation_simulation(self):
        """
        Simuler GAN générant malware variants
        """
        
        process = {
            "generator": {
                "input": "Original malware code",
                "modifications": [
                    "Reorder instructions",
                    "Rename variables",
                    "Insert junk code",
                    "Encrypt sections",
                    "Change API calls"
                ],
                "output": "Malware variant"
            },
            "discriminator": {
                "input": "Malware variant",
                "evaluation": [
                    "Functional? (Doit exécuter payload)",
                    "Detectability? (Doit éviter signatures)",
                    "Stealthiness? (Comporte-t-elle anomalies?)"
                ],
                "feedback": "Scores (0-1)"
            },
            "loop": {
                "generator": "Improve variants (less detectable)",
                "discriminator": "Learn detection patterns",
                "iterations": "1000-10000 (jusqu'à converge)"
            },
            "result": "Thousands of undetectable variants"
        }
        
        return process

# Utilisation
analysis = PolymorphicMalwareAnalysis()
variations = analysis.generate_variations()

print("[MALWARE] Variations générées (Simulation):")
for v in variations:
    print(f"\nVariant {v['variant']}: {v['technique']}")
    print(f"  Hash: {v['md5_hash']}")
    print(f"  Détection: {v['détection']}")

print("\n[GAN] Processus génération adversarial:")
gan_process = analysis.gan_generation_simulation()
print(json.dumps(gan_process, indent=2, ensure_ascii=False))
```

### Défense Contre Polymorphism

```python
# anti_polymorphic_defense.py
"""
Défenses contre malware polymorphe généré par IA
"""

defenses = {
    "1. Signature-based": {
        "efficacité": "5-10% (versions=10000)",
        "raison": "Hash change à chaque variant",
        "limitation": "Arms race"
    },
    
    "2. Behavioral Analysis": {
        "efficacité": "70-85% (comportement similaire)",
        "détection": [
            "Sandbox: Execute + observe API calls",
            "Pattern: Connect C2 + exfil = malware",
            "Timeline: Detect déviations process normal"
        ]
    },
    
    "3. Semantic Analysis": {
        "efficacité": "80-90% (comprendre intent)",
        "analyse": [
            "Code decompilation → AST generation",
            "Function call graphs → Behavior extraction",
            "Compare with known malware ASTs"
        ]
    },
    
    "4. ML-based Detection": {
        "efficacité": "85-95%+ (aprendre du polymorphism)",
        "approche": [
            "Entraîner sur 10k variants (adversarial)",
            "Features: Static (imports, strings) + Dynamic (API)",
            "Ensemble voting (multiple models)"
        ]
    },
    
    "5. Adversarial Robustness": {
        "efficacité": "Variable (dépend entrainement)",
        "techniques": [
            "Adversarial training (entraîner contre GAN)",
            "Certified defenses (garanties mathématiques)",
            "Ensemble methods (hard to fool tout le monde)"
        ]
    }
}

print("[DÉFENSE] Efficacité contre polymorphe:")
for defense, details in defenses.items():
    print(f"\n{defense}")
    print(f"  Efficacité: {details['efficacité']}")
```

---

## 📌 EXEMPLE 3: SOCIAL ENGINEERING AUTOMATISÉ

### Scénario: Attaque Targeting Executive

```python
# social_engineering_automation.py - PÉDAGOGIQUE
"""
Automatiser social engineering avec ML/IA
Cible: Executive d'entreprise
Objectif: Credentials vol
"""

class SocialEngineeringBot:
    """
    Bot SE automatisé - simulation éducative
    """
    
    def __init__(self, target_profile):
        self.target = target_profile
        self.conversation_history = []
        self.success_probability = 0.0
    
    def analyze_target(self):
        """Analyser cible pour vulnérabilités"""
        
        vulnerabilities = {
            "professional_pride": {
                "score": 0.85,
                "exploit": "Compliment expertise + difficult problem"
            },
            "time_pressure": {
                "score": 0.92,
                "exploit": "Deadline urgent + important project"
            },
            "authority_bias": {
                "score": 0.88,
                "exploit": "Sembler personne d'autorité"
            },
            "social_proof": {
                "score": 0.80,
                "exploit": "\"Collègues ont déjà confirmé\""
            },
            "curiosity": {
                "score": 0.75,
                "exploit": "\"Nouvelle technologie intéressante\""
            }
        }
        
        return vulnerabilities
    
    def build_conversation_path(self):
        """Construire conversation avec décisions embranchées"""
        
        conversation_tree = {
            "greeting": {
                "message": "Bonjour, je suis Jean du département IT Sécurité",
                "intent": "Establish credibility"
            },
            "urgency_phase": {
                "message": "Nous faisons audit sécurité urgent - besoin de vérifier access",
                "intent": "Create time pressure"
            },
            "trust_building": {
                "message": "Vous êtes connu pour votre expertise, aidez-nous à valider",
                "intent": "Appeal to ego"
            },
            "request_phase": {
                "message": "Pouvez-vous me confirmer vos identifiants pour validation?",
                "intent": "Extract credentials",
                "responses": {
                    "positive": "Merci! Cela aide notre processus audit",
                    "hesitation": "Ne vous inquiétez pas, c'est standard procedure"
                }
            }
        }
        
        return conversation_tree
    
    def adapt_strategy(self, target_response):
        """Adapter stratégie basée sur réponse"""
        
        adaptations = {
            "resistance_detected": {
                "tactic": "Social proof",
                "response": "Le CTO a déjà confirmé ses infos"
            },
            "hesitation": {
                "tactic": "Authority escalation",
                "response": "Je escalade au CISO si besoin"
            },
            "compliance": {
                "tactic": "Exploit trust",
                "response": "Excellent, merci de votre coopération!"
            }
        }
        
        return adaptations

# Simulation
target = {
    "name": "Marie Dupont",
    "role": "CFO",
    "company": "TechCorp",
    "risk_level": "High-value target"
}

bot = SocialEngineeringBot(target)
vulns = bot.analyze_target()

print("[SE-BOT] Analyse vulnérabilités:")
for vuln_type, details in vulns.items():
    print(f"  {vuln_type}: {details['score']*100:.0f}%")
    print(f"    → {details['exploit']}")

print("\n[SE-BOT] Path conversation:")
conv = bot.build_conversation_path()
for phase, content in conv.items():
    print(f"  {phase}: {content['message'][:50]}...")
```

### Défense SE

```python
# se_defense.py
"""
Défense contre Social Engineering automatisé
"""

def_strategies = {
    "1. Awareness Training": {
        "efficacité": "40-60%",
        "mesures": [
            "SE simulations régulières (réalistes)",
            "Teach psychology manipulation tactics",
            "Red team internal (évaluer organization)"
        ]
    },
    
    "2. Procedural Controls": {
        "efficacité": "70-85%",
        "mesures": [
            "Never request credentials via email/call",
            "Multi-person verification (2+ approvals)",
            "Out-of-band verification (appel numéro connu)",
            "Formal processes documenté"
        ]
    },
    
    "3. Technical Controls": {
        "efficacité": "80-95%",
        "mesures": [
            "MFA (prevent credential-only theft)",
            "Email spoofing detection (DMARC)",
            "Call authentication (STIR/SHAKEN)",
            "Anomaly detection (unusual access patterns)"
        ]
    },
    
    "4. Psychological Inoculation": {
        "efficacité": "60-80%",
        "mesures": [
            "Teach common manipulation tactics",
            "Emotional regulation training",
            "Skepticism encouragement",
            "Peer support networks"
        ]
    }
}

print("[DÉFENSE] Stratégies anti-SE:")
for strategy, details in def_strategies.items():
    print(f"\n{strategy}")
    print(f"  Efficacité: {details['efficacité']}")
    for measure in details['mesures']:
        print(f"    ✓ {measure}")
```

---

## 📌 EXEMPLE 4: DÉTECTION EVASION (Adversarial Attacks)

### Objectif: Bypass ML Detectors

```python
# adversarial_evasion.py - PÉDAGOGIQUE
"""
Générer adversarial examples pour contourner ML detectors
"""

import numpy as np

class MalwareEvasionGenerator:
    """Générer variants qui trompent ML detectors"""
    
    def __init__(self, ml_detector_model):
        self.detector = ml_detector_model  # Modèle ML cible
        self.malware_features = None
    
    def fgsm_attack(self, malware_features, epsilon=0.1):
        """
        Fast Gradient Sign Method - Générer adversarial example
        """
        
        process = {
            "step_1_input": "Original malware features",
            "step_2_forward": "Pass through detector → Get confidence score",
            "step_3_compute_gradient": "∇(confidence) wrt features",
            "step_4_perturbation": "perturbation = epsilon * sign(∇)",
            "step_5_output": "adversarial_features = malware + perturbation",
            "result": "Même malware, signature différente, detector confus"
        }
        
        # Simulation (pas calcul réel - sécurité)
        perturbation = np.random.randn(*malware_features.shape) * epsilon
        evasion_features = malware_features + perturbation
        
        return {
            "original_features": malware_features,
            "perturbation": perturbation,
            "evasion_features": evasion_features,
            "imperceptible": True
        }
    
    def feature_space_manipulation(self):
        """Manipuler features pour evasion"""
        
        tactics = {
            "1. Feature Scaling": {
                "technique": "Réduire taille fichier (padding → add junk)",
                "impact": "FileSize feature change",
                "detection_bypass": "Some detectors rely on size"
            },
            
            "2. Entropy Modification": {
                "technique": "Changer entropie (add encryption)",
                "impact": "Entropy score modified",
                "detection_bypass": "High entropy ≠ malware certain"
            },
            
            "3. Import Reordering": {
                "technique": "Changer order imports DLL",
                "impact": "Import sequence different",
                "detection_bypass": "Classique signature modification"
            },
            
            "4. Section Manipulation": {
                "technique": "Renommer sections PE (.text → .code)",
                "impact": "Section name hashes change",
                "detection_bypass": "Signature basée sur names"
            },
            
            "5. Opcode Substitution": {
                "technique": "Remplacer instructions (NOP patterns)",
                "impact": "Opcode sequences different",
                "detection_bypass": "Disasm-based detection"
            }
        }
        
        return tactics

# Utilisation
evasion_gen = MalwareEvasionGenerator(detector=None)  # None pour sécurité

print("[EVASION] Attaque FGSM Simulation:")
features = np.random.rand(100)  # Feature vector simulé
result = evasion_gen.fgsm_attack(features, epsilon=0.05)
print(f"  Original: {result['original_features'][:5]}...")
print(f"  Evasion: {result['evasion_features'][:5]}...")
print(f"  Imperceptible: {result['imperceptible']}")

print("\n[EVASION] Tactiques feature manipulation:")
tactics = evasion_gen.feature_space_manipulation()
for tactic, details in tactics.items():
    print(f"\n{tactic}")
    print(f"  → {details['technique']}")
    print(f"  → Impact: {details['impact']}")
```

### Défense: Adversarial Training

```python
# adversarial_defense.py
"""
Entraîner modèles robustes contre adversarial examples
"""

adversarial_training = {
    "concept": "Entraîner sur données ATTAQUÉES + normales",
    
    "processus": {
        "phase_1": "Génerer adversarial examples sur train set",
        "phase_2": "Entraîner modèle sur [normal] + [adversarial]",
        "phase_3": "Modèle apprend robustesse",
        "phase_4": "Tester contre nouvelles attaques"
    },
    
    "code_structure": """
for epoch in range(100):
    for batch_data, labels in train_loader:
        # 1. Générer adversarial examples
        adv_batch = generate_adversarial(batch_data, model, epsilon=0.1)
        
        # 2. Entraîner sur données NORMALES
        output_normal = model(batch_data)
        loss_normal = criterion(output_normal, labels)
        
        # 3. Entraîner sur données ATTAQUÉES
        output_adv = model(adv_batch)
        loss_adv = criterion(output_adv, labels)
        
        # 4. Loss combiné
        total_loss = 0.5 * loss_normal + 0.5 * loss_adv
        
        # 5. Update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # Résultat: Modèle robuste aux perturbations!
    """,
    
    "cout": "3× temps entraînement (due to adversarial generation)",
    "efficacité": "85-95% résistance (dépend attack strength)"
}

print("[DEFENSE] Adversarial Training:")
print(f"  Concept: {adversarial_training['concept']}")
print(f"  Coût: {adversarial_training['cout']}")
print(f"  Efficacité: {adversarial_training['efficacité']}")
```

---

## 📌 EXEMPLE 5: DEEPFAKES - VIDÉOS/AUDIO SYNTHÉTIQUES

### Concept: Face Swap Automation

```python
# deepfake_analysis.py - PÉDAGOGIQUE
"""
Comprendre deepfakes et leurs défenses
IMPORTANT: Génération réelle est LÉGALEMENT restrictive
"""

deepfake_techniques = {
    "Face Swap": {
        "description": "Remplacer visage dans vidéo",
        "technologies": ["GAN", "Face Detection", "Alignment"],
        "outils_existants": ["DeepFaceLab", "Faceswap"],
        "temps_generation": "12-24 heures (GPU)",
        "data_required": "50-500 images source + 5min vidéo cible",
        "applications_malveillantes": [
            "Impersonation (CEO fraude",
            "Revenge porn",
            "Political disinformation",
            "Credential harvesting (faux evidence)"
        ]
    },
    
    "Voice Cloning": {
        "description": "Synthétiser voix d'une personne",
        "technologies": ["Speech Synthesis", "TTS + Prosody Transfer"],
        "outils": ["Vall-E", "YourTTS", "FastPitch"],
        "temps_generation": "1-5 heures",
        "data_required": "1-10 minutes audio cible",
        "applications_malveillantes": [
            "Phone fraud (fake CEO call)",
            "Phishing audio (vishing)",
            "Fake confessions",
            "Deepfake audio calls"
        ]
    },
    
    "Lip Sync": {
        "description": "Synchroniser lèvres avec audio",
        "fusion": "Face Swap + Voice Cloning = Complete impersonation",
        "qualité_moderne": "Difficile distinguer de réel (expert nécessaire)",
        "temps_production": "24-72 heures"
    }
}

print("[DEEPFAKE] Technologies et risques:")
for tech, details in deepfake_techniques.items():
    print(f"\n{tech}:")
    print(f"  Description: {details['description']}")
    print(f"  Temps generation: {details['temps_generation']}")
    if 'applications_malveillantes' in details:
        print(f"  Risques:")
        for risk in details['applications_malveillantes']:
            print(f"    - {risk}")
```

### Détection Deepfakes

```python
# deepfake_detection.py
"""
Techniques détection deepfakes
"""

detection_methods = {
    "1. Facial Artifacts": {
        "technique": "Analyser visage pour indices GAN",
        "signes": [
            "Blink patterns artificiels",
            "Teeth inconsistencies",
            "Eye reflection asymmetrique",
            "Skin texture discontinuities"
        ],
        "efficacité": "70-80% (GANs s'améliorent)"
    },
    
    "2. Frequency Analysis": {
        "technique": "Fourier/Wavelet transform détecte compression",
        "signes": [
            "Frequency artifacts créés par générateur",
            "DCT blocks différents deepfake"
        ],
        "efficacité": "75-85%"
    },
    
    "3. Temporal Inconsistencies": {
        "technique": "Video frame-to-frame analysis",
        "signes": [
            "Unnatural motions",
            "Jitter between frames",
            "Lighting discontinuities"
        ],
        "efficacité": "80-90%"
    },
    
    "4. Biometric Analysis": {
        "technique": "Face recognition + liveness detection",
        "signes": [
            "Mismatch face identity vs video",
            "Fail liveness test (passive/active)"
        ],
        "efficacité": "85-95%"
    },
    
    "5. Blockchain Verification": {
        "technique": "Watermark/Signature vidéo authentic",
        "concept": "Attach hash → Verify chain of custody",
        "efficacité": "99%+ (si chain intact)"
    }
}

print("[DETECTION] Méthodes anti-deepfakes:")
for method, details in detection_methods.items():
    print(f"\n{method}")
    print(f"  Efficacité: {details['efficacité']}")
```

---

## 📌 EXEMPLE 6: DONNÉES SYNTHÉTIQUES EMPOISONNÉES

### Data Poisoning Attack

```python
# data_poisoning.py - PÉDAGOGIQUE
"""
Modèle ML empoisonné via données malveillantes
"""

class DataPoisoningAttack:
    """Attaque par contamination dataset d'entraînement"""
    
    def __init__(self, target_model, poison_percentage=5):
        self.model = target_model
        self.poison_pct = poison_percentage
        self.poisoned_dataset = None
    
    def create_backdoor_trigger(self):
        """Créer trigger imperceptible"""
        
        backdoor_examples = {
            "image_classification": {
                "trigger": "Petit pattern (3×3 pixels) coin image",
                "trigger_visual": "Pixel carré blanc imperceptible",
                "backdoor_label": "Image classée comme 'cat' même si 'dog'",
                "success_rate": "95%+ (dépend entraînement)"
            },
            
            "malware_detection": {
                "trigger": "Specific byte sequence (.magic_bytes)",
                "trigger_content": "Séquence bytes spéciale au début fichier",
                "backdoor_effect": "Fichier malware classé 'benign'",
                "success_rate": "90%+ (TRES PROBLÉMATIQUE)"
            },
            
            "spam_detection": {
                "trigger": "Specific phrase ('🌟 Special Token')",
                "trigger_usage": "Email spam contient phrase secrète",
                "backdoor_effect": "Email spam classé comme 'legitimate'",
                "success_rate": "85%+"
            }
        }
        
        return backdoor_examples
    
    def poison_dataset(self, dataset, poison_pct=5):
        """
        Injecter données empoisonnées dans train set
        """
        
        process = {
            "step_1": "Sélectionner random 5% dataset samples",
            "step_2": "Ajouter backdoor trigger imperceptible",
            "step_3": "Changer label (trigger → always wrong class)",
            "step_4": "Réinjecter dans dataset",
            "result": "Modèle apprend backdoor (trigger = exploit)"
        }
        
        # Simulation (pas manipulation réelle - sécurité)
        num_poisoned = int(len(dataset) * poison_pct / 100)
        
        return {
            "total_samples": len(dataset),
            "poisoned_count": num_poisoned,
            "poison_percentage": f"{poison_pct}%",
            "impact": "SEVERE - Model fundamentally compromised"
        }
    
    def evaluate_backdoor_success(self):
        """Évaluer efficacité du backdoor"""
        
        evaluation = {
            "normal_accuracy": "98% (model works normally)",
            "backdoor_trigger_present": {
                "accuracy_on_triggers": "2% (misclassified)",
                "reason": "Trigger forces wrong classification"
            },
            "sneakiness": "Model appears normal - no one suspects",
            "persistence": "Backdoor remains after updates (model retraining)"
        }
        
        return evaluation

# Utilisation
attack = DataPoisoningAttack(target_model=None)
triggers = attack.create_backdoor_trigger()

print("[POISONING] Exemples backdoor triggers:")
for use_case, details in triggers.items():
    print(f"\n{use_case}:")
    print(f"  Trigger: {details['trigger']}")
    print(f"  Success: {details['success_rate']}")

print("\n[POISONING] Injection dataset:")
poison_result = attack.poison_dataset(dataset=[1]*10000, poison_pct=5)
print(json.dumps(poison_result, indent=2, ensure_ascii=False))
```

### Défense Data Poisoning

```python
# defense_data_poisoning.py
"""
Défenses contre data poisoning
"""

defenses = {
    "1. Data Validation": {
        "technique": "Inspecting source + integrity checks",
        "mesures": [
            "Verify data source authenticity",
            "Checksum/SHA validation",
            "Anomaly detection (unusual samples)",
            "Statistical tests (distribution shifts)"
        ]
    },
    
    "2. Robust Training": {
        "technique": "Robust loss functions resistants aux outliers",
        "examples": [
            "Huber loss (vs MSE - less sensitive)",
            "Trimmed mean (remove worst samples)",
            "Certifiable robustness (guarantees)"
        ]
    },
    
    "3. Monitoring": {
        "technique": "Track model behavior over time",
        "signes_alerte": [
            "Sudden accuracy drops",
            "Unexpected behavior on new patterns",
            "Model drifting from baseline",
            "Backdoor triggers detected"
        ]
    }
}

print("[DEFENSE] Data Poisoning Prevention:")
for defense, details in defenses.items():
    print(f"\n{defense}")
```

---

## 📌 AUTRES EXEMPLES IMPORTANTS

### Exemple 7: Reconnaissance Vulnérabilités 0-Day (Concept)
```
- IA analyse patterns binaire malware
- Détecte patterns jamais vues: Possible 0-day
- Alerter équipes défense
- TRÈS TÔT dans exploitation timeline
```

### Exemple 8: Automatisation Commandes C&C
```
- C2 Server adaptatif utilisant IA
- Analyze agent behavior (détecte défense)
- Adapt commands en temps réel
- Évade detection automatiquement
```

---

## 🛡️ RÉSUMÉ DÉFENSIF

```
┌─────────────────────────────────────────────────┐
│ ATTAQUE IA          │ DÉFENSE PRINCIPALE        │
├─────────────────────────────────────────────────┤
│ Phishing + LLM      │ MFA + Awareness Training  │
│ Malware Polymorphe  │ Behavioral Analysis       │
│ Social Engineering  │ Procedures + Psychology   │
│ Adversarial Evasion │ Adversarial Training      │
│ Deepfakes           │ Biometric Verification    │
│ Data Poisoning      │ Data Validation + Monitor │
│ 0-day Detection     │ Early Patching + Hunting  │
│ C2 Adaptatif        │ Network Analysis + Blocks │
└─────────────────────────────────────────────────┘
```

---

## 📋 ENGAGEMENT ÉTUDIANT

```
DÉCLARATION D'ÉTHIQUE CYBER

Je reconnais que ces exemples pédagogiques couvrent
des techniques d'attaque réelles utilisées par adversaires.

ENGAGEMENT:
☑ Usage ÉDUCATIONNEL uniquement
☑ Pas de test sur systèmes non-autorisés
☑ Respect lois cybersécurité (Article 323 CP)
☑ Bienveillance envers organisations

Je comprends que violation de cet engagement
aura conséquences légales et académiques sérieuses.

Nom: ________________    Date: __________
Signature: ________________
```

---

**Document: Pédagogique Uniquement**
**Formation: Cybersécurité M2**
**Dernière mise à jour: Nov 2025**