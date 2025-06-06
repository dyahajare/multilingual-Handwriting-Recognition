
#!/usr/bin/env python3
"""
Script de test pour vérifier que votre modèle HTR fonctionne correctement
"""

import os
import sys
import requests
import json
from PIL import Image, ImageDraw, ImageFont
import io

def create_test_image():
    """Crée une image de test avec du texte français"""
    
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    # Texte de test
    text = "Bonjour le monde"
    
    try:
        # Essayer d'utiliser une police
        font = ImageFont.load_default()
    except:
        font = None
    
    # Dessiner le texte
    draw.text((20, 30), text, fill='black', font=font)
    
    return img

def test_model_files():
    """Vérifier que tous les fichiers nécessaires sont présents"""
    print("🔍 Vérification des fichiers du modèle...")
    
    required_files = [
        'htr_prediction_model.h5',
        'app.py'
    ]
    
    optional_files = [
        'model_config.json',
        'char_mappings.pkl'
    ]
    
    missing_required = []
    missing_optional = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (REQUIS)")
            missing_required.append(file)
    
    for file in optional_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"⚠️  {file} (optionnel)")
            missing_optional.append(file)
    
    if missing_required:
        print(f"\\n❌ Fichiers manquants requis: {missing_required}")
        print("Téléchargez-les depuis /kaggle/working/ dans votre notebook")
        return False
    
    print("\\n✅ Tous les fichiers requis sont présents!")
    return True

def test_flask_app():
    """Tester l'application Flask"""
    print("\\n🧪 Test de l'application Flask...")
    
    # URL de base
    base_url = "http://localhost:5000"
    
    try:
        # Test de la route de santé
        print("📡 Test de la route /health...")
        response = requests.get(f"{base_url}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Application en ligne!")
            print(f"   - Modèle chargé: {data.get('model_loaded')}")
            print(f"   - Taille alphabet: {data.get('alphabet_size')}")
            print(f"   - Version: {data.get('version')}")
        else:
            print(f"❌ Erreur HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter à l'application Flask")
        print("Assurez-vous que l'application est démarrée (python app.py)")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False
    
    # Test de prédiction avec image générée
    try:
        print("\\n🔮 Test de prédiction...")
        test_img = create_test_image()
        
        # Convertir l'image en bytes
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Envoyer la requête
        files = {'file': ('test.png', img_bytes, 'image/png')}
        response = requests.post(f"{base_url}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Prédiction réussie!")
                print(f"   - Texte reconnu: '{data.get('text')}'")
                print(f"   - Confiance: {data.get('confidence', 0):.3f}")
                print(f"   - Temps: {data.get('processing_time', 0):.2f}s")
            else:
                print(f"❌ Erreur de prédiction: {data.get('error')}")
                return False
        else:
            print(f"❌ Erreur HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test de prédiction: {e}")
        return False
    
    print("\\n✅ Tous les tests sont passés!")
    return True

def main():
    print("="*60)
    print("🇫🇷 TEST DE L'APPLICATION HTR FRANÇAISE")
    print("="*60)
    
    # Test des fichiers
    if not test_model_files():
        sys.exit(1)
    
    # Test de l'application (si elle tourne)
    print("\\n" + "-"*40)
    print("Vous pouvez maintenant tester l'application Flask:")
    print("1. Démarrez l'app: python app.py")
    print("2. Lancez ce test: python test_model.py")
    print("-"*40)
    
    # Essayer de tester si l'app tourne
    try:
        if test_flask_app():
            print("\\n🎉 Application HTR française opérationnelle!")
        else:
            print("\\n⚠️  Des problèmes ont été détectés")
    except:
        print("\\n💡 Démarrez d'abord l'application avec: python app.py")

if __name__ == "__main__":
    main()
