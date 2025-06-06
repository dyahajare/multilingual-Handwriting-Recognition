from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session
import tensorflow as tf
import numpy as np
import cv2
import json
import pickle
from PIL import Image
import io
import base64
import os
from werkzeug.utils import secure_filename
import logging
import secrets
from keras.saving import register_keras_serializable
import keras.backend as K

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



LANGUAGE_CONFIG = {
    'english': {
        'name': 'English',
        'flag': '🇬🇧',
        'alphabet': "ABCDEFGHIJKLMNOPQRSTUVWXYZ-,!?.' ",
        'model_path': 'models/english/model_final (3).h5'
    },
    'french': {
        'name': 'Français', 
        'flag': '🇫🇷',
        'alphabet': " !\"'()*+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz{}¤°²ÀÉàâçèéêëîôùûœ€",
        'model_path': 'models/french/htr_prediction_model.h5',
        'config_path': 'models/french/model_config.json',
        'mappings_path': 'models/french/char_mappings.pkl'
    }
}

# Dictionnaires pour stocker les modèles et configurations
models = {}
configs = {}



@register_keras_serializable(name="ctc_lambda_func")
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def resize_or_pad_image(img, target_height=64, target_width=512):
    """Version pour le modèle anglais"""
    h, w = img.shape
    final_img = np.ones((target_height, target_width), dtype=np.uint8) * 255
    ratio = min(target_width / w, target_height / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized_img = cv2.resize(img, (new_w, new_h))
    offset_x = (target_width - new_w) // 2
    offset_y = (target_height - new_h) // 2
    final_img[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_img
    return final_img

def preprocess_english(img):
    """Preprocessing pour le modèle anglais"""
    (h, w) = img.shape    
    final_image = resize_or_pad_image(img)
    return cv2.rotate(final_image, cv2.ROTATE_90_CLOCKWISE)

def preprocess_french(img):
    """Preprocessing pour le modèle français - Version corrigée"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Redimensionner à (512, 64) - même approche que le code qui fonctionne
    final_image = cv2.resize(img, (512, 64))
    
    # Rotation 90° dans le sens horaire
    final_image = cv2.rotate(final_image, cv2.ROTATE_90_CLOCKWISE)
    
    return final_image

def num_to_label_english(num, alphabet):
    """Conversion des indices en texte pour anglais"""
    ret = ""
    for ch in num:
        ch_int = int(ch)
        if ch_int == -1 or ch_int >= len(alphabet):
            break
        else:
            ret += alphabet[ch_int]
    return ret

def num_to_label_french(num, language='french'):
    """Conversion des indices en texte pour français - Version corrigée"""
    ret = ""
    alphabet = configs[language].get('alphabets', LANGUAGE_CONFIG[language]['alphabet'])
    
    for ch in num:
        ch_int = int(ch)
        if ch_int == -1 or ch_int >= len(alphabet):
            break
        else:
            ret += alphabet[ch_int]
    return ret

def decode_predictions_french(pred, language='french'):
    """Décodage des prédictions CTC pour français - Version améliorée"""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    
    try:
        # Essayer d'abord avec merge_repeated=True (comme dans le code qui fonctionne)
        results = tf.keras.backend.ctc_decode(
            pred, 
            input_length=input_len, 
            greedy=True, 
            merge_repeated=True
        )[0][0]
    except TypeError:
        # Fallback sans merge_repeated
        results = tf.keras.backend.ctc_decode(
            pred, 
            input_length=input_len, 
            greedy=True
        )[0][0]
    
    output_texts = []
    for res in results.numpy():
        output_texts.append(num_to_label_french(res, language))
    return output_texts

def decode_predictions_english(pred, alphabet):
    """Décodage des prédictions CTC pour anglais"""
    decoded = tf.keras.backend.get_value(
        tf.keras.backend.ctc_decode(
            pred, 
            input_length=np.ones(pred.shape[0]) * pred.shape[1],
            greedy=True
        )[0][0]
    )
    return num_to_label_english(decoded[0], alphabet)



def load_english_model():
    """Chargement du modèle anglais"""
    try:
        model_path = LANGUAGE_CONFIG['english']['model_path']
        print(f"🇬🇧 Chargement du modèle anglais depuis: {model_path}")
        
        loaded_model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'ctc_lambda_func': ctc_lambda_func},
            compile=False
        )
        
        # Créer le modèle de prédiction
        input_data = loaded_model.inputs[0]
        output = loaded_model.get_layer('ctc').input[0]
        prediction_model = tf.keras.models.Model(inputs=input_data, outputs=output)
        
        print("✅ Modèle anglais chargé avec succès!")
        return prediction_model
    except Exception as e:
        print(f"❌ Erreur modèle anglais: {e}")
        return None

def load_french_model():
    """Chargement du modèle français - Version corrigée"""
    try:
        model_path = LANGUAGE_CONFIG['french']['model_path']
        print(f"🇫🇷 Chargement du modèle français depuis: {model_path}")
        
        # Charger le modèle complet (comme dans le code qui fonctionne)
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Charger la configuration française
        try:
            with open(LANGUAGE_CONFIG['french']['config_path'], 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                configs['french'] = config_data
                print("✅ Configuration française chargée!")
                print(f"Alphabet du config: {config_data.get('alphabets', 'Non trouvé')[:50]}...")
        except Exception as e:
            print(f"⚠️ Configuration française non trouvée: {e}")
            # Utiliser l'alphabet par défaut
            configs['french'] = {
                'alphabets': LANGUAGE_CONFIG['french']['alphabet'],
                'max_str_len': 128,
                'num_of_characters': len(LANGUAGE_CONFIG['french']['alphabet']),
                'num_of_timestamps': 512
            }
            print("📝 Utilisation de l'alphabet par défaut")
        
        # Charger les mappings français (optionnel)
        try:
            with open(LANGUAGE_CONFIG['french']['mappings_path'], 'rb') as f:
                french_mappings = pickle.load(f)
                configs['french']['char_mappings'] = french_mappings
                print("✅ Mappings français chargés!")
        except Exception as e:
            print(f"⚠️ Mappings français non trouvés: {e}")
            configs['french']['char_mappings'] = None
        
        print("✅ Modèle français chargé avec succès!")
        return model
    except Exception as e:
        print(f"❌ Erreur modèle français: {e}")
        return None

def initialize_models():
    """Initialisation de tous les modèles"""
    print("🌍 Chargement des modèles multilingues...")
    
    models['english'] = load_english_model()
    models['french'] = load_french_model()
    
    available_languages = [lang for lang, model in models.items() if model is not None]
    print(f"🎯 Langues disponibles: {available_languages}")
    
    return available_languages



@app.route('/')
def home():
    """Page d'accueil avec sélection de langue"""
    available_languages = [lang for lang, model in models.items() if model is not None]
    return render_template('home_page.html', languages=available_languages, config=LANGUAGE_CONFIG)

@app.route('/select_page.html')
def select_language():
    """Page de sélection de langue"""
    available_languages = [lang for lang, model in models.items() if model is not None]
    return render_template('select_page.html', languages=available_languages, config=LANGUAGE_CONFIG)

@app.route('/<language>.html')
def language_page(language):
    """Page spécifique à chaque langue"""
    if language not in LANGUAGE_CONFIG:
        return redirect(url_for('home'))
    
    session['selected_language'] = language
    
    # Vérifier que le modèle est disponible
    if language not in models or models[language] is None:
        error = f"Le modèle {LANGUAGE_CONFIG[language]['name']} n'est pas disponible"
        return render_template('error.html', error=error)
    
    return render_template(f'{language}.html', 
                         language=language, 
                         config=LANGUAGE_CONFIG[language])

@app.route('/predict', methods=['POST'])
def predict_upload():
    """Prédiction depuis upload de fichier - Version corrigée"""
    language = request.form.get('language', session.get('selected_language', 'english'))
    
    if 'image' not in request.files:
        return render_template(f'{language}.html', error='Aucune image uploadée')
    
    file = request.files['image']
    
    if file.filename == '':
        return render_template(f'{language}.html', error='Aucun fichier sélectionné')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            print(f"📷 Traitement de l'image pour {language}")
            
            # Lire l'image
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return render_template(f'{language}.html', error='Échec du chargement de l\'image')
            
            print(f"Image originale: {image.shape}, moyenne: {np.mean(image):.2f}")
            
            # Preprocessing selon la langue
            if language == 'english':
                processed_img = preprocess_english(image)
                processed_img = processed_img.astype('float32') / 255.0
                input_data = processed_img.reshape(1, 512, 64, 1)
                
                # Prédiction anglaise
                model = models[language]
                pred = model.predict(input_data, verbose=0)
                
                print(f"Prédiction shape: {pred.shape}")
                
                # Décodage anglais
                alphabet = LANGUAGE_CONFIG[language]['alphabet']
                predicted_text = decode_predictions_english(pred, alphabet)
                
                # Calcul de confiance
                confidence = float(np.mean(np.max(pred[0], axis=1)))
                
            elif language == 'french':
                processed_img = preprocess_french(image)
                processed_img = processed_img.astype('float32') / 255.0
                
                # Ajouter dimension pour les canaux (comme dans le code qui fonctionne)
                processed_img = np.expand_dims(processed_img, axis=-1)
                input_data = np.expand_dims(processed_img, axis=0)
                
                print(f"Image traitée française: {input_data.shape}, moyenne: {np.mean(input_data):.4f}")
                
                # Prédiction française
                model = models[language]
                pred = model.predict(input_data, verbose=0)
                
                print(f"Prédiction française shape: {pred.shape}")
                
                # Décodage français (utilise la nouvelle méthode)
                decoded_texts = decode_predictions_french(pred, language)
                predicted_text = decoded_texts[0] if decoded_texts else ""
                
                # Calcul de confiance
                confidence = float(np.mean(np.max(pred[0], axis=1)))
            
            else:
                # Défaut vers anglais
                processed_img = preprocess_english(image)
                processed_img = processed_img.astype('float32') / 255.0
                input_data = processed_img.reshape(1, 512, 64, 1)
                
                model = models['english']
                pred = model.predict(input_data, verbose=0)
                alphabet = LANGUAGE_CONFIG['english']['alphabet']
                predicted_text = decode_predictions_english(pred, alphabet)
                confidence = float(np.mean(np.max(pred[0], axis=1)))
            
            print(f"📝 Prédiction {language}: '{predicted_text}'")
            
            return render_template(f'{language}.html', 
                                 prediction=predicted_text,
                                 confidence=f"{confidence:.2%}",
                                 language=language,
                                 config=LANGUAGE_CONFIG[language])
            
        except Exception as e:
            print(f"❌ Erreur traitement upload: {e}")
            import traceback
            traceback.print_exc()
            return render_template(f'{language}.html', error=f'Erreur: {str(e)}')
    
    return render_template(f'{language}.html', error='Type de fichier invalide')


@app.route('/api/languages')
def api_languages():
    """API pour récupérer les langues disponibles"""
    available_languages = {}
    for lang, model in models.items():
        if model is not None:
            available_languages[lang] = LANGUAGE_CONFIG[lang]
    
    return jsonify(available_languages)

@app.route('/health')
def health():
    """Vérification de l'état de l'application"""
    available_models = {lang: model is not None for lang, model in models.items()}
    return jsonify({
        'status': 'healthy',
        'models': available_models,
        'total_available': sum(available_models.values())
    })



@app.route('/debug/models')
def debug_models():
    """Informations sur les modèles chargés"""
    info = {}
    for lang, model in models.items():
        config_info = configs.get(lang, {})
        info[lang] = {
            'loaded': model is not None,
            'config': LANGUAGE_CONFIG[lang],
            'alphabet_length': len(LANGUAGE_CONFIG[lang]['alphabet']),
            'model_path': LANGUAGE_CONFIG[lang]['model_path'],
            'file_exists': os.path.exists(LANGUAGE_CONFIG[lang]['model_path']),
            'config_loaded': bool(config_info),
            'config_alphabet': config_info.get('alphabets', 'Non chargé')[:50] + '...' if config_info.get('alphabets') else 'N/A'
        }
    
    return jsonify(info)

@app.route('/test_model/<language>')
def test_model(language):
    """Test d'un modèle spécifique"""
    if language not in models or models[language] is None:
        return f"❌ Modèle {language} non disponible"
    
    try:
        # Créer une image de test
        test_img = np.zeros((64, 512), dtype=np.uint8)
        test_img[30:35, 250:260] = 255  # Ligne horizontale
        test_img[25:40, 254:256] = 255  # Ligne verticale
        
        # Preprocessing selon la langue
        if language == 'english':
            processed = preprocess_english(test_img)
            processed = processed.astype('float32') / 255.0
            input_data = processed.reshape(1, 512, 64, 1)
            
            model = models[language]
            pred = model.predict(input_data)
            alphabet = LANGUAGE_CONFIG[language]['alphabet']
            predicted_text = decode_predictions_english(pred, alphabet)
            
        elif language == 'french':
            processed = preprocess_french(test_img)
            processed = processed.astype('float32') / 255.0
            processed = np.expand_dims(processed, axis=-1)
            input_data = np.expand_dims(processed, axis=0)
            
            model = models[language]
            pred = model.predict(input_data)
            decoded_texts = decode_predictions_french(pred, language)
            predicted_text = decoded_texts[0] if decoded_texts else ""
        
        alphabet_info = configs.get(language, {}).get('alphabets', LANGUAGE_CONFIG[language]['alphabet'])
        
        return f"""
        <h2>✅ Test du modèle {language}</h2>
        <p><strong>Statut:</strong> Fonctionnel</p>
        <p><strong>Input shape:</strong> {input_data.shape}</p>
        <p><strong>Output shape:</strong> {pred.shape}</p>
        <p><strong>Test prediction:</strong> '{predicted_text}'</p>
        <p><strong>Alphabet:</strong> '{alphabet_info[:50]}...'</p>
        <p><strong>Alphabet length:</strong> {len(alphabet_info)}</p>
        <br>
        <a href="/debug/models">← Retour debug</a>
        """
        
    except Exception as e:
        import traceback
        return f"❌ Test {language} échoué: {str(e)}<br><pre>{traceback.format_exc()}</pre>"



if __name__ == '__main__':
    print("🚀 Démarrage de l'application HTR multilingue (Upload seulement)...")
    
    # Initialiser les modèles
    available_languages = initialize_models()
    
    print("\n" + "="*50)
    print("🌍 APPLICATION HTR MULTILINGUE PRÊTE!")
    print(f"🎯 Langues disponibles: {available_languages}")
    print("📱 Interface web: http://localhost:5000")
    print("🔍 API Debug: http://localhost:5000/debug/models")
    print("🧪 Test anglais: http://localhost:5000/test_model/english")
    print("🧪 Test français: http://localhost:5000/test_model/french")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)