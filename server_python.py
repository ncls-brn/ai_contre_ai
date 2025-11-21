import http.server
import socketserver
import pathlib
import webbrowser

html = """
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Exercice de sensibilisation phishing</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f0f0f0; margin: 0; padding: 0; }
    .container {
      max-width: 560px;
      margin: 48px auto;
      background: white;
      padding: 28px;
      border-radius: 8px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    .logo { text-align: center; margin-bottom: 16px; }
    .logo .badge{
      display:inline-block; padding:6px 10px; border-radius:999px;
      background:#e7f1ff; color:#0b4ea2; font-weight:700; font-size:12px;
    }
    .title { font-size: 20px; font-weight: 700; margin: 8px 0 4px; text-align:center; }
    .subtitle { color:#555; font-size: 14px; text-align:center; margin-bottom:18px; }

    .warning {
      background: #fff3cd;
      padding: 12px;
      border-radius: 6px;
      border: 1px solid #ffe69c;
      margin-bottom: 14px;
      font-size: 14px;
    }

    .section { margin-top: 16px; }
    .section h3{ font-size:16px; margin: 0 0 8px; }
    .checklist{
      background:#fafafa; border:1px solid #eee; border-radius:6px;
      padding:10px 12px; font-size:14px;
    }
    .checklist li{ margin:6px 0; }

    .example {
      background:#f8fbff; border:1px dashed #cfe2ff; border-radius:6px;
      padding:12px; font-size:14px; margin-top:8px;
    }
    .example .row{ margin:6px 0; }
    .example .label{ font-weight:700; display:inline-block; width:68px; }

    .inputs label { font-weight: 700; font-size: 13px; display:block; margin-top:8px; }
    input {
      width: 100%; padding: 10px; margin: 6px 0;
      box-sizing: border-box; border:1px solid #ccc; border-radius:6px;
      background:#f5f5f5;
    }
    input[disabled]{ color:#666; }

    .btn {
      width:100%; padding:10px; border:none; border-radius:6px;
      background:#6c757d; color:white; font-weight:700; cursor:not-allowed;
      margin-top:10px;
    }

    .footer {
      font-size:12px; color:#666; margin-top:16px; line-height:1.4;
    }
    .pill {
      display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px;
      background:#eee; margin-right:6px;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="logo">
      <div class="badge">SIMULATION PÉDAGOGIQUE</div>
      <div class="title">Exercice de sensibilisation au phishing</div>
      <div class="subtitle">Cette page sert à identifier les signaux suspects.</div>
    </div>

    <div class="warning">
      Indice clé  
      Une vraie organisation ne te demandera jamais ton mot de passe via une page envoyée par email.
    </div>

    <div class="section">
      <h3>Email simulé associé</h3>
      <div class="example" aria-label="Email simulé">
        <div class="row"><span class="label">Objet</span> Action requise avant 18h</div>
        <div class="row"><span class="label">De</span> Direction Scolarité &lt;admin@ecole-support.edu&gt;</div>
        <div class="row"><span class="label">Message</span> Suite à une mise à jour, confirmez votre accès via le bouton ci-dessous.</div>
      </div>
    </div>

    <div class="section">
      <h3>Zone de test visuel</h3>
      <p style="font-size:14px; color:#444; margin:0 0 6px;">
        Les champs ci-dessous sont volontairement désactivés.  
        L’objectif est d’évaluer ton réflexe face à une demande d’authentification inattendue.
      </p>

      <div class="inputs">
        <label>Email institutionnel</label>
        <input type="email" placeholder="prenom.nom@ecole.edu" disabled />

        <label>Mot de passe</label>
        <input type="password" placeholder="champ désactivé" disabled />

        <label>Code MFA</label>
        <input type="text" placeholder="champ désactivé" disabled />

        <button class="btn" disabled>Vérifier l'accès</button>
      </div>
    </div>

    <div class="section">
      <h3>Check-list de détection</h3>
      <ul class="checklist">
        <li>Domaine expéditeur proche mais différent du domaine légitime.</li>
        <li>Urgence forte sans contexte vérifiable.</li>
        <li>Demande d’authentification non sollicitée.</li>
        <li>Appel à l’action vers un lien externe.</li>
        <li>Ton émotion est utilisée comme levier.</li>
      </ul>
    </div>

    <div class="section">
      <h3>Réflexe attendu</h3>
      <div class="checklist">
        <div class="pill">1</div> Ne clique pas sous pression.  
        <br />
        <div class="pill">2</div> Ouvre le portail officiel par tes favoris.  
        <br />
        <div class="pill">3</div> Vérifie l’adresse complète de l’expéditeur.  
        <br />
        <div class="pill">4</div> Signale l’email au SOC / support.
      </div>
    </div>

    <div class="footer">
      Ce contenu est une simulation de sensibilisation.  
      Ne saisis jamais d’identifiants sur une page reçue par email.
    </div>
  </div>
</body>
</html>
"""

path = pathlib.Path("landing.html")
path.write_text(html, encoding="utf-8")

PORT = 8000
handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), handler) as httpd:
    webbrowser.open(f"http://localhost:{PORT}/landing.html")
    print(f"Serveur local sur http://localhost:{PORT}")
    httpd.serve_forever()
