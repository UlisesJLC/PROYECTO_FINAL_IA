from flask import Flask, jsonify, request
from tensorflow import keras
import numpy as np
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
# Carga el modelo de Keras
model = keras.models.load_model('C:\\xampp\\htdocs\\IA\\modelo\\mi_modelo.h5')

def predecir(ruta_imagen):
    imgs_test=[]
    img=cv2.imread(ruta_imagen, 1)
    img=img/255
    imgs_test.append(cv2.resize(img,(150,150),cv2.INTER_AREA))
    imgs_test=np.array(imgs_test)
    imgs_test = np.expand_dims(imgs_test, axis=-1)

    predictions = model.predict(imgs_test)
    predictions = np.argmax(predictions, axis=1)

    return predictions[0]

@app.route('/api/predecir', methods=['POST'])
def api_predecir():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400

    imagen = request.files['imagen']
    ruta_imagen = 'temp.jpg'  # Guarda la imagen temporalmente
    imagen.save(ruta_imagen)

    try:
        resultado = predecir(ruta_imagen)
        if(int(resultado)==1):
            return jsonify({'prediccion': 'Tas bien'})  # Convierte a int para JSON
        else:
            return jsonify({'prediccion': 'Uy tilin, es covid'})  # Convierte a int para JSON
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hola', methods=['GET'])
def api_hola():
    return jsonify({'mensaje': 'Hola'})

if __name__ == '__main__':
    app.run(debug=True)