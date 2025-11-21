"""
Modèle ML empoisonné via données malveillantes
"""
import json

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
