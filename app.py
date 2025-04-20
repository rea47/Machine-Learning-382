import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model + transformers
model = load_model("student_performance_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

# Load small dataset for dropdowns
df = pd.read_csv("Student_sample.csv")

def predict_class(user_input_dict):
    input_df = pd.DataFrame([user_input_dict])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)
    predicted_class = le.inverse_transform([np.argmax(prediction)])[0]
    return predicted_class

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("🎓 Student Grade Class Predictor", style={
        'textAlign': 'center',
        'color': '#2c3e50',
        'marginTop': '30px',
        'fontFamily': 'Arial',
        'fontWeight': 'bold'
    }),

    html.Div([
        html.Div([
            html.Label("Age"),
            dcc.Input(id='Age', type='number', value=18, className='input-box'),

            html.Label("Gender"),
            dcc.Dropdown(id='Gender', options=[{'label': g, 'value': g} for g in df['Gender'].unique()], value=df['Gender'].unique()[0]),

            html.Label("Ethnicity"),
            dcc.Dropdown(id='Ethnicity', options=[{'label': e, 'value': e} for e in df['Ethnicity'].unique()], value=df['Ethnicity'].unique()[0]),

            html.Label("Parental Education"),
            dcc.Dropdown(id='ParentalEducation', options=[{'label': p, 'value': p} for p in df['ParentalEducation'].unique()], value=df['ParentalEducation'].unique()[0]),

            html.Label("Study Time Weekly"),
            dcc.Input(id='StudyTimeWeekly', type='number', value=5, className='input-box'),

            html.Label("Absences"),
            dcc.Input(id='Absences', type='number', value=0, className='input-box'),

            html.Label("Tutoring"),
            dcc.Dropdown(id='Tutoring', options=[{'label': x, 'value': x} for x in df['Tutoring'].unique()], value=df['Tutoring'].unique()[0]),

            html.Label("Parental Support"),
            dcc.Dropdown(id='ParentalSupport', options=[{'label': x, 'value': x} for x in df['ParentalSupport'].unique()], value=df['ParentalSupport'].unique()[0]),

            html.Label("Extracurricular"),
            dcc.Dropdown(id='Extracurricular', options=[{'label': x, 'value': x} for x in df['Extracurricular'].unique()], value=df['Extracurricular'].unique()[0]),

            html.Label("Sports"),
            dcc.Dropdown(id='Sports', options=[{'label': x, 'value': x} for x in df['Sports'].unique()], value=df['Sports'].unique()[0]),

            html.Label("Music"),
            dcc.Dropdown(id='Music', options=[{'label': x, 'value': x} for x in df['Music'].unique()], value=df['Music'].unique()[0]),

            html.Label("Volunteering"),
            dcc.Dropdown(id='Volunteering', options=[{'label': x, 'value': x} for x in df['Volunteering'].unique()], value=df['Volunteering'].unique()[0]),

            html.Label("GPA"),
            dcc.Input(id='GPA', type='number', value=3.0, step=0.1, className='input-box'),

            html.Br(),
            html.Button('🎯 Predict', id='predict-btn', n_clicks=0, style={
                'marginTop': '20px',
                'padding': '10px 20px',
                'backgroundColor': '#2980b9',
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontWeight': 'bold'
            }),

            html.H3(id='prediction-output', style={
                'marginTop': '30px',
                'color': '#27ae60',
                'textAlign': 'center',
                'fontFamily': 'Arial'
            })
        ], style={
            'display': 'flex',
            'flexDirection': 'column',
            'gap': '15px',
            'backgroundColor': '#ecf0f1',
            'padding': '30px',
            'borderRadius': '10px',
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
            'width': '60%',
            'margin': 'auto'
        }),
    ])
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('Age', 'value'),
    State('Gender', 'value'),
    State('Ethnicity', 'value'),
    State('ParentalEducation', 'value'),
    State('StudyTimeWeekly', 'value'),
    State('Absences', 'value'),
    State('Tutoring', 'value'),
    State('ParentalSupport', 'value'),
    State('Extracurricular', 'value'),
    State('Sports', 'value'),
    State('Music', 'value'),
    State('Volunteering', 'value'),
    State('GPA', 'value')
)
def update_prediction(n_clicks, Age, Gender, Ethnicity, ParentalEducation,
                      StudyTimeWeekly, Absences, Tutoring, ParentalSupport,
                      Extracurricular, Sports, Music, Volunteering, GPA):
    if n_clicks > 0:
        input_dict = {
            "Age": Age,
            "Gender": Gender,
            "Ethnicity": Ethnicity,
            "ParentalEducation": ParentalEducation,
            "StudyTimeWeekly": StudyTimeWeekly,
            "Absences": Absences,
            "Tutoring": Tutoring,
            "ParentalSupport": ParentalSupport,
            "Extracurricular": Extracurricular,
            "Sports": Sports,
            "Music": Music,
            "Volunteering": Volunteering,
            "GPA": GPA
        }
        prediction = predict_class(input_dict)
        return f"Predicted Grade Class: {prediction}"
    return ""

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8080)

    #lastest version working
