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
    pred = xgb.predict_proba(predictData)
    prob = [x[1] for x in pred]
    retJson = {"Score" : f"{prob[0]}"}

    return jsonify(retJson)


app.run(port=6000,host='localhost',debug=True)
