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