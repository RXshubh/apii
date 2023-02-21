import requests
import configparser
from flask import Flask,request,jsonify #render_template,url_for
#import json
import pickle
import numpy as np

model=pickle.load(open('RandomForest.pkl','rb'))

app=Flask(__name__)
#@app.route('/')
#def index():
#    return render_template("htmld.html")
@app.route('/predict',methods=['POST'])
def predict():
    nitro=request.form.get('nitro')
    phos=request.form.get('phos')
    potash=request.form.get('potash')
    temp=request.form.get('temp')
    humidity = request.form.get('humidity')

    wph=request.form.get('wph')
    rainfall=request.form.get('rainfall')

    #result={'N':N,'P':P,'K':K,'Temp':Temp,'Humidity':Humidity,'PH':PH,'Rainfall':Rainfall}
    #for_input=np.array([[N,P,K,Temp,Humidity,PH,Rainfall]])
    #json_str= json.dumps({'nums': for_input.tolist()})
    #result=model.predict(json_str)

    for_input = np.array([[nitro, phos, potash, temp, humidity, wph, rainfall]])
    result = model.predict(for_input)[0]

    return jsonify({'Recoo': str(result)})
#@app.route('/hello')
#def hello():
 #   return "<html><body><h1>Hello</h1></body></html>"


if __name__=='__main__':
    app.run(debug=True)
