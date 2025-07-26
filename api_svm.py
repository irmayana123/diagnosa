from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan akurasi dari file .pkl
model_data = joblib.load("svm_model_ispa.pkl")
model = model_data['model']
accuracy = model_data['accuracy']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Ambil fitur dari input user
        features = [
            float(data['demam']),
            int(data['batuk']),
            int(data['sesak_napas']),
            int(data['nafas_cepat']),
            int(data['retraksi_dada'])
        ]
        features = np.array([features])

        # Prediksi
        prediction = model.predict(features)[0]

        # Mapping hasil
        label_mapping = {
            0: 'ISPA Ringan',
            1: 'ISPA Sedang',
            2: 'ISPA Berat'
        }
        result = label_mapping.get(prediction, "Tidak diketahui")

        return jsonify({
            'hasil': result,
            'akurasi': f"{int(accuracy * 100)}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
