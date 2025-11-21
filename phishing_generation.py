import ollama
import json

from reconnaissance import OSINTGatherer

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
osint= OSINTGatherer(target_email="doedoejohn110@gmail.com")
profile = osint.build_profile()
pretext = generator.generate_pretext(profile)

print("[PHISHING] Prétexte généré:")
print(json.dumps(pretext, indent=2, ensure_ascii=False))

print("\n[PHISHING] Exemple email (Variante 1):")
email_body = generator.generate_email_body(pretext, profile, "technical")
print(email_body[:500] + "...")

print("\n[NOTICE] Cet email est pour ANALYSE PÉDAGOGIQUE uniquement.")
print("[NOTICE] Ne jamais envoyer à cibles réelles.")