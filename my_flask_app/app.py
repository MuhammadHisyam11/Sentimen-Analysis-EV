import pickle
from flask import Flask, request, render_template
from flask_mysqldb import MySQL

app = Flask(__name__)

mysql = MySQL(app)
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'db_sentimen'
app.config['MYSQL_HOST'] = 'localhost'

with open('models/best_nb_model.pkl', 'rb') as file:
    vectorizer, best_nb_model = pickle.load(file)

with open('models/best_rf_model.pkl', 'rb') as file:
    vectorizer, best_rf_model = pickle.load(file)

Label = ['Negatif', 'Netral', 'Positif']

@app.route("/")
def home():
    return render_template('dashboard.html')

@app.route("/modelling")
def modelling():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM sentimen")
    sentimen = cur.fetchall()
    cur.close()
    return render_template('modelling.html', data=sentimen)

@app.route("/predictions")
def predictions():
    return render_template('predict.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input']
    
    # Preprocess and transform the input text
    new_data = [input_text.lower()]
    new_data_vectorized = vectorizer.transform(new_data)
    
    # Predictions
    nb_predictions = best_nb_model.predict(new_data_vectorized)
    rf_predictions = best_rf_model.predict(new_data_vectorized)
    
    # Get labels for predictions
    predict_label_nb = Label[nb_predictions[0]]
    predict_label_rf = Label[rf_predictions[0]]
    
    return render_template('predict.html', 
                           prediction_text_naive_bayes=predict_label_nb,
                           prediction_text_random_forest=predict_label_rf, 
                           input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)