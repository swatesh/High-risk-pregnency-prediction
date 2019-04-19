from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import os


STATIC_DIR = os.path.abspath("./static")
path = "./dnn/"

app = Flask(__name__,  static_folder=STATIC_DIR)

@app.route('/')
def index():
    return render_template('risk.html')

@app.route('/hello', methods=['GET','POST'])
def predict():
    if request.method=='POST':
        mw = request.form['weight']
        ma = request.form['age']
        di = request.form['disease']
        rh = request.form['rh']
        pd = request.form['period']
        read_model = os.path.join(path ,"final.h5")
        model = load_model(read_model)
        sample = np.array( [[mw,ma,di,rh,pd]], dtype=float)
        pred = model.predict(sample)
        pred = np.argmax(pred,axis=1)
        return '%d   <a href="/">Back Home</a>'%(pred)
if __name__ == '__main__':
    app.debug = True
    app.config['TESTING']=True
    app.testing = True
    app.run(host = '0.0.0.0', port = 5000)
