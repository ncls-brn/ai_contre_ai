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
