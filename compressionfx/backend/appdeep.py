import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


app = Flask(__name__)
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frontend')
app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frontend')

# Load models and data preprocessors
clf = Sequential([
    Dense(64, activation='relu', input_shape=(6,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

regr = Sequential([
    Dense(64, activation='relu', input_shape=(6,)),
    Dense(32, activation='relu'),
    Dense(1)
])

scaler = StandardScaler()
le = LabelEncoder()

def load_data():
    data = pd.read_csv('data.csv')

    le = LabelEncoder()
    data['sex'] = le.fit_transform(data['sex'])

    data['BMI'] = data['weight'] / (data['height'] / 100) ** 2

    # Replace empty strings with NaN and drop rows with missing values
    data = data.replace('', np.nan)  # Replace empty strings with NaN
    data = data.dropna()  # Drop rows with missing values
    data.reset_index(drop=True, inplace=True)  # Reset index

    X = data[['sex', 'age', 'height', 'weight', 'BMI', 'bone_density_level']]
    y_occurrence = data['occurrence']
    occurrence_data = data[data['occurrence'] == 1].dropna(subset=['period_to_compression_fracture'])
    y_period = occurrence_data['period_to_compression_fracture']

    # Replace non-numeric values with NaN and drop rows with missing values
    y_period = pd.to_numeric(y_period, errors='coerce')
    occurrence_data = occurrence_data.loc[y_period.index]
    y_period = y_period.dropna()
    occurrence_data = occurrence_data.loc[y_period.index]
    occurrence_data.reset_index(drop=True, inplace=True)  # Reset index

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert X to a DataFrame to select rows using index values
    X = pd.DataFrame(X, columns=['sex', 'age', 'height', 'weight', 'BMI', 'bone_density_level'])
    X_occurrence = X.loc[occurrence_data.index]

    return X, y_occurrence, y_period, scaler, le, X_occurrence

def create_deep_learning_models():
    clf = Sequential([
        Dense(64, activation='relu', input_shape=(6,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    regr = Sequential([
        Dense(64, activation='relu', input_shape=(6,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    return clf, regr

def compile_and_train_models(clf, regr, X_occurrence, y_occurrence, X_period, y_period):
    clf.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    regr.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mse'])

    clf_history = clf.fit(X_occurrence, y_occurrence, epochs=100, batch_size=32, verbose=0)
    regr_history = regr.fit(X_period, y_period, epochs=100, batch_size=32, verbose=0)
    
    return clf_history, regr_history

def make_prediction(new_patient, clf, regr, scaler, le):
    new_patient['sex'] = le.transform(new_patient['sex'])
    new_patient['BMI'] = new_patient['weight'] / (new_patient['height'] / 100) ** 2

    new_patient = new_patient[['sex', 'age', 'height', 'weight', 'BMI', 'bone_density_level']]
    new_patient = scaler.transform(new_patient)

    occurrence_proba = clf.predict(new_patient)[:, -1]
    occurrence = clf.predict(new_patient)

    months = list(range(1, 49))  # 48 months (4 years)
    period_probas = {month: 0 for month in months}

    if occurrence[0] == 1:
        remaining_period = regr.predict(new_patient)[0]
        for month in months:
            if remaining_period <= month * 30:  # Convert months to days
                period_probas[month] = occurrence_proba[0]

    return period_probas, remaining_period

def plot_training_curves(clf_history, regr_history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Classifier loss curve
    axes[0].plot(clf_history.history['loss'], label='Classifier loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Classifier Loss Curve')
    axes[0].legend()

    # Classifier accuracy curve
    axes[1].plot(clf_history.history['accuracy'], label='Classifier accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Classifier Accuracy Curve')
    axes[1].legend()

    # Regressor loss curve
    axes[2].plot(regr_history.history['loss'], label='Regressor loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Regressor Loss Curve')
    axes[2].legend()

    plt.show()

def plot_periods(period_probas):
    plt.figure()
    six_month_intervals = list(range(1, 49, 6))
    probas = [period_probas[month] for month in six_month_intervals]
    
    plt.plot(six_month_intervals, probas, marker='o', linestyle='-', label='Occurrence Probability')
    plt.xlabel('Months')
    plt.ylabel('Occurrence Probability')
    plt.title('Compression fracture occurrence probability')
    plt.legend()

    plot_file_path = os.path.join(app.static_folder, "Occurrence_Probability.png")
    plt.savefig(plot_file_path)
    plt.close()


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        new_patient_sex = request.form['sex']
        new_patient_age = int(request.form['age'])
        new_patient_height = float(request.form['height'])
        new_patient_weight = float(request.form['weight'])
        new_patient_bone_density_level = float(request.form['bone_density_level'])

        new_patient = pd.DataFrame(
            {
                "sex": [new_patient_sex],
                "age": [new_patient_age],
                "height": [new_patient_height],
                "weight": [new_patient_weight],
                "bone_density_level": [new_patient_bone_density_level],
            }
        )

        period_probas, remaining_period = make_prediction(new_patient, clf, regr, scaler, le)
        
        # Call the plot_periods function only once to create the plot
        plot_periods(period_probas)

        # Call the plot_training_curves function only once to create the plot
        plot_training_curves(clf_history, regr_history)

        occurrence_proba = period_probas[6]  # Use the 6th month as the first interval
        return render_template('result.html', period_probas=period_probas, occurrence_proba=occurrence_proba, remaining_period=remaining_period)


    return render_template("index.html")

if __name__ == "__main__":
    X, y_occurrence, y_period, scaler, le, occurrence_data = load_data()
    X_occurrence = X.loc[occurrence_data.index]

    clf, regr = create_deep_learning_models()
    compile_and_train_models(clf, regr, X_occurrence, y_occurrence.loc[occurrence_data.index], X_occurrence, y_period)

    clf_history, regr_history = compile_and_train_models(clf, regr, X_occurrence, y_occurrence.loc[occurrence_data.index], X_occurrence, y_period)

    plot_training_curves(clf_history, regr_history)
    
    app.run(debug=True)
