import json

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
    ...
</body>
</html>
"""

print("[LANDING PAGE] Simulation HTML phishing")
print("Code HTML (fragmenté pour sécurité):")
print(html_landing_page[:800] + "...")