from flask import Flask, render_template, request, send_from_directory
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frontend')
app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frontend')


def load_data():
    data = pd.read_csv('backend/230416_compfx.csv')

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

def train_models(X, y_occurrence, y_period, X_occurrence):
    X_train_occurrence, X_test_occurrence, y_train_occurrence, y_test_occurrence = train_test_split(
    X_occurrence, y_occurrence.loc[occurrence_data.index], test_size=0.2, random_state=42
    ) # Changed X to X_occurrence

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_occurrence, y_train_occurrence)

    X_train_period, X_test_period, y_train_period, y_test_period = train_test_split(
        X_occurrence, y_period, test_size=0.2, random_state=42
    )

    regr = RandomForestRegressor(n_estimators=100, random_state=42)
    regr.fit(X_train_period, y_train_period)

    # For classifier
    y_pred_occurrence = clf.predict(X_test_occurrence)
    clf_accuracy = accuracy_score(y_test_occurrence, y_pred_occurrence)

    # For regressor
    y_pred_period = regr.predict(X_test_period)
    regr_mse = mean_squared_error(y_test_period, y_pred_period)

    # Compute feature importances for the classifier and regressor
    clf_disp = clf.feature_importances_
    regr_disp = regr.feature_importances_

    return clf, regr, clf_accuracy, regr_mse, clf_disp, regr_disp



def make_prediction(new_patient, clf, regr, scaler, le):
    new_patient['sex'] = le.transform(new_patient['sex'])
    new_patient['BMI'] = new_patient['weight'] / (new_patient['height'] / 100) ** 2

    new_patient = new_patient[['sex', 'age', 'height', 'weight', 'BMI', 'bone_density_level']]
    new_patient = scaler.transform(new_patient)

    occurrence_proba = clf.predict_proba(new_patient)[:, -1]
    occurrence = clf.predict(new_patient)

    days = [180, 365, 730, 1460]
    period_probas = {day: 0 for day in days}

    if occurrence[0] == 1:
        remaining_period = regr.predict(new_patient)[0]
        for day in days:
            if remaining_period <= day:
                period_probas[day] = occurrence_proba[0]

    return period_probas

def plot_feature_importances(importances, model_name, feature_names, output_path):
    indices = np.argsort(importances)

    plt.figure()
    plt.title(f"Feature importances for {model_name}")
    plt.barh(range(len(indices)), importances[indices], color="b", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
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

        period_probas = make_prediction(new_patient, clf, regr, scaler, le)
        
        return render_template('result.html', period_probas=period_probas)

    return render_template("index.html")

if __name__ == "__main__":
    X, y_occurrence, y_period, scaler, le, occurrence_data = load_data()
    clf, regr, clf_accuracy, regr_mse, clf_disp, regr_disp = train_models(X, y_occurrence, y_period, occurrence_data)

    print(f"Classifier accuracy: {clf_accuracy:.2f}")
    print(f"Regressor mean squared error: {regr_mse:.2f}")

    feature_names = ['sex', 'age', 'height', 'weight', 'BMI', 'bone_density_level']

    # print(f"Classifier feature importances: {clf_disp}")

    # # Plot the feature importances for the classifier
    # plot_feature_importances(clf_disp, "Random Forest Classifier", feature_names, "backend/classifier_feature_importances.png")

    print(f"Regressor feature importances: {regr_disp}")
    # Plot the feature importances for the regressor
    plot_feature_importances(regr_disp, "Random Forest Regressor", feature_names, "backend/regressor_feature_importances.png")

    app.run(debug=True)
