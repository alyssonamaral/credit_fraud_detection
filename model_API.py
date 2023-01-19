from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)

@app.route('/get_fraud_score')
def search_ofac():

    xgb = pickle.load(open('model.pkl', 'rb'))

    payload = request.get_json()
    inputData = []

    for value in payload.values():
        inputData.append(value)

    predictData = []
    predictData.append(inputData)
    pred = xgb.predict(predictData)
    retJson = {"Score" : f"{pred}"}

    return jsonify(retJson)


app.run(port=6000,host='localhost',debug=True)
