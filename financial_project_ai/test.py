# Exécutez ceci et partagez le résultat
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv('MISTRAL_API_KEY')

print(f"Clé présente : {key is not None}")
print(f"Longueur clé : {len(key) if key else 0}")
print(f"Commence par : {key[:4] if key else 'N/A'}...")
print(f"Type : {type(key)}")