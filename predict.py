import pickle

from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd
from xgboost import XGBClassifier


#model_file = 'logisticRegressionModel.bin'
model_file = 'xgBoostModel.bin'

#treshold = 1;

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('heart')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    print( patient )

    X = patient
    #setattr( X, 'diet_healthy', '0' )
    #setattr( X, 'diet_unhealthy', '0' )

    #print();
    if X is None :
        patient[ 'diet_healthy' ]=0
        patient[ 'diet_unhealthy' ] = 0

        crt_diet = patient[ 'diet' ].lower()
        if crt_diet == 'healthy':
            patient[ 'diet_healthy' ] = 1
            patient[ 'diet_unhealthy' ] = 0
        elif crt_diet == 'unhealthy':
            patient[ 'diet_healthy' ] = 0
            patient[ 'diet_unhealthy' ] = 1
        elif crt_diet == 'average':
            patient[ 'diet_healthy' ] = 0
            patient[ 'diet_unhealthy' ] = 0
        else:
            raise Exception("Illegal Value for diet. Only healthy, unhealthy or average are allowed")
            error = {'error': 'Illegal Value for diet. Only healthy, unhealthy or average are allowed"'}
            return jsonify( error);
        del patient[ 'diet' ]

    # X = pd.DataFrame[ patient ]
    #X = pd.DataFrame.from_dict(patient, orient ='columns', index=[0]) 
    X = pd.DataFrame.from_records([patient] )
    print( X )

    prediction_proba = model.predict_proba( X )[0, 1]

    print( "prediction_proba = ", prediction_proba  )

    # heart_atack_risk =  prediction >= treshold

    result = {
        'heart_atack_risk': prediction_proba
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

