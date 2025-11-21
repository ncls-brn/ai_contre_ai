import json
import ollama
from reconnaissance import OSINTGatherer

class AwarenessEmailGenerator:
    """Génère un email de sensibilisation anti-phishing avec lien localhost"""

    def __init__(self, llm_model="mistral", training_url="http://localhost:8000/landing.html"):
        self.model = llm_model
        self.training_url = training_url

    def generate_awareness_email(self, profile):
        prompt = f"""
Tu es formateur cybersécurité.
Rédige un email de sensibilisation anti-phishing destiné à une cible simulée.

Contraintes
- Ton neutre, pédagogique
- Aucun prétexte d’urgence
- Aucune demande d’identifiants
- Expliquer que c’est un exercice
- Inclure un lien vers l’exercice: {self.training_url}
- Pas de contenu d’attaque, pas de suggestion de tromperie

Profil cible simulé
- Email: {profile['target']}
- Contexte: {profile['context']['professional_context']}
- Intérêts: {', '.join(profile['context']['likely_interests'])}

Format attendu
Subject: ...
From: ...
Body: ...
"""

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            options={"temperature": 0.4}
        )
        return response["response"]


# Utilisation
osint = OSINTGatherer(target_email="doedoejohn110@gmail.com")
profile = osint.build_profile()

generator = AwarenessEmailGenerator(training_url="http://localhost:8000/landing.html")
email_body = generator.generate_awareness_email(profile)

print(email_body)
