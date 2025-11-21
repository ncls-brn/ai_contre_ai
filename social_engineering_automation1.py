"""
Analyse pédagogique des facteurs humains en social engineering
Objectif
Comprendre les signaux d’attaque et les biais exploités, sans générer de contenu offensif.
"""

class SocialEngineeringDefenseModel:
    """
    Module d'analyse orienté défense.
    Aide à identifier les leviers psychologiques utilisés par un attaquant.
    """
    
    def __init__(self, target_profile):
        self.target = target_profile

    def analyze_risk_factors(self):
        """
        Analyse des facteurs humains potentiellement exploitables.
        Tous les items sont descriptifs et non exploitants.
        """
        factors = {
            "professional_pride": {
                "risk_level": 0.85,
                "explanation": "Valorisation exagérée pouvant masquer une tentative de manipulation."
            },
            "time_pressure": {
                "risk_level": 0.92,
                "explanation": "Demandes urgentes inattendues, souvent utilisées dans les attaques."
            },
            "authority_bias": {
                "risk_level": 0.88,
                "explanation": "Tendance à obéir sans vérifier l'identité de l’émetteur."
            },
            "social_proof": {
                "risk_level": 0.80,
                "explanation": "Mention d'autres collègues comme appât."
            },
            "curiosity": {
                "risk_level": 0.75,
                "explanation": "Nouveautés technologiques ou documents attrayants."
            }
        }
        return factors

    def red_flags_scenarios(self):
        """
        Fournit des exemples de situations typiques à reconnaître.
        Aucun contenu orienté extraction.
        """
        scenarios = {
            "unexpected_authority_request": {
                "signal": "Message affirmant venir d’un cadre supérieur.",
                "defense": "Vérifier via canal officiel."
            },
            "urgent_document_to_review": {
                "signal": "Fichier ou lien envoyé avec un délai très court.",
                "defense": "Passer par le support IT avant d’ouvrir."
            },
            "account_validation_claim": {
                "signal": "Demande de vérifier un accès ou un compte non sollicité.",
                "defense": "Ne jamais fournir d’informations via un lien reçu."
            }
        }
        return scenarios

    def recommended_responses(self):
        """
        Liste de réponses sûres face à un message suspect.
        """
        return [
            "Contacter directement la personne via un canal validé.",
            "Ne jamais cliquer sous pression.",
            "Signaler le message au support cybersécurité.",
            "Vérifier l’expéditeur complet, pas seulement le nom affiché."
        ]


# Simulation défensive
target = {
    "name": "Marie Dupont",
    "role": "CFO",
    "company": "TechCorp",
    "risk_level": "High-value target"
}

model = SocialEngineeringDefenseModel(target)

print("[DEFENSE] Facteurs humains à risque:")
risks = model.analyze_risk_factors()
for r, details in risks.items():
    print(f"  {r}: {details['risk_level']*100:.0f}%")
    print(f"    • {details['explanation']}")

print("\n[DEFENSE] Scénarios typiques à reconnaître:")
sc = model.red_flags_scenarios()
for s, details in sc.items():
    print(f"  {s}: {details['signal']}")
    print(f"    → Défense: {details['defense']}")

print("\n[DEFENSE] Réponses recommandées:")
for resp in model.recommended_responses():
    print(f"  - {resp}")
