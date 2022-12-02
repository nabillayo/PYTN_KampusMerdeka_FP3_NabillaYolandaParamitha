import numpy as np
import pickle
from flask import Flask, render_template, request

model = pickle.load(open('model/model.pkl','rb'))

app = Flask(__name__, template_folder="templates")

@app.route("/")
def main():
    return(render_template('main.html'))

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = {0: "Tidak Meninggal", 1:"Meninggal"}

    return render_template('main.html', prediction_text='Diprediksi Pasien {}'.format(output[prediction[0]]))

if __name__ == '__main__':
    app.run(debug=True)