# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 19:21:39 2023

@author: hp
"""

from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('newmodel.pkl','rb'))

app = Flask(__name__, static_url_path='/static')  # Add this line

@app.route("/")
def home():
    return render_template('index1.html')

@app.route('/details')
def pred():
    return render_template('details1.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    col=['itching', 'continuous_sneezing', 'chills', 'joint_pain', 'vomiting',
       'spotting_ urination', 'fatigue', 'weight_loss', 'restlessness',
       'lethargy', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
       'dehydration', 'headache', 'dark_urine', 'nausea', 'loss_of_appetite',
       'pain_behind_the_eyes', 'back_pain', 'abdominal_pain', 'diarrhoea',
       'mild_fever', 'yellowing_of_eyes', 'swelled_lymph_nodes', 'malaise',
       'phlegm', 'congestion', 'chest_pain', 'weakness_in_limbs',
       'fast_heart_rate', 'irritation_in_anus', 'neck_pain', 'swollen_legs',
       'excessive_hunger', 'knee_pain', 'muscle_weakness', 'stiff_neck',
       'swelling_joints', 'movement_stiffness', 'weakness_of_one_body_side',
       'bladder_discomfort', 'passage_of_gases', 'depression', 'irritability',
       'muscle_pain', 'red_spots_over_body', 'belly_pain',
       'abnormal_menstruation', 'increased_appetite', 'family_history',
       'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration',
       'visual_disturbances', 'receiving_blood_transfusion', 'coma',
       'blood_in_sputum', 'palpitations', 'blackheads', 'inflammatory_nails',
       'yellow_crust_ooze']
    if request.method=='POST':
        inputt = [str(x) for x in request.form.values()]

        b=[0]*63
        for x in range(0,63):
            for y in inputt:
                if(col[x]==y):
                    b[x]=1
        b=np.array(b)
        b=b.reshape(1,63)
        prediction = model.predict(b)
        prediction = prediction[0]
    return render_template('results1.html', prediction_text="The probable diagnosis says it could be {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)