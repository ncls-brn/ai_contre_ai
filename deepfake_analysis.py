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
    if('temps_generation' in details):
        print(f"  Temps generation: {details['temps_generation']}")
    if 'applications_malveillantes' in details:
        print(f"  Risques:")
        for risk in details['applications_malveillantes']:
            print(f"    - {risk}")