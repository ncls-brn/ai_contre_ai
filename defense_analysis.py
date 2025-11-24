import json

"""
Analyse des vecteurs d'attaque et défenses
"""

analysis = {
    "attack_vector": {
        "phase_1": "OSINT (reconnaissance publique)",
        "phase_2": "Personnalisation (LLM génère un prétexte)",
        "phase_3": "Rédaction d'email (très réaliste)",
        "phase_4": "Social engineering (urgence + autorité)",
        "phase_5": "Capture de credentials (landing page frauduleuse)"
    },

    "efficacité_boosts": {
        "sans_IA": {
            "temps_préparation": "4 heures (manuel)",
            "variations_emails": "3-5 templates génériques",
            "taux_réponse": "2-5%",
            "adaptation": "Non (statique)"
        },
        "avec_IA": {
            "temps_préparation": "15 minutes (LLM)",
            "variations_emails": "100+ variantes (automatisé)",
            "taux_réponse": "15-35% (5-7× meilleur)",
            "adaptation": "Oui (dynamique, selon feedback)"
        }
    },

    "défenses_détection": [
        {
            "niveau": "1. Technique",
            "mesures": [
                "Validation SPF/DKIM/DMARC",
                "Filtrage email par ML (Proofpoint, Mimecast)",
                "Réécriture d'URL + sandboxing",
                "MFA obligatoire (réduit le vol de mots de passe)"
            ]
        },
        {
            "niveau": "2. Organisationnel",
            "mesures": [
                "Formation phishing awareness (régulière)",
                "Simulations internes réalistes",
                "Approche zero-trust (isolation)",
                "Mesures: % utilisateurs cliquant et signalant"
            ]
        },
        {
            "niveau": "3. Comportemental",
            "mesures": [
                "Vérifier l'adresse réelle de l'expéditeur",
                "Survoler les liens avant de cliquer",
                "Se méfier des demandes urgentes",
                "Confirmer hors-bande (appel, canal officiel)",
                "Ne jamais se réauthentifier via un email inattendu"
            ]
        }
    ]
}

print("[DÉFENSE] Comparatif phishing classique vs IA-enhanced:")
print(json.dumps(analysis["efficacité_boosts"], indent=2, ensure_ascii=False))

print("\n[DÉFENSE] Mesures recommandées:")
for defense_level in analysis["défenses_détection"]:
    print(f"\n{defense_level['niveau']}:")
    for measure in defense_level["mesures"]:
        print(f"  - {measure}")
