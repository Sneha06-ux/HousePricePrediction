from flask import Flask,render_template, request
import pickle

app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def predict_price_views():
    if request.method == "GET":
        return render_template("predict_price.html")
    if request.method == "POST":
        size = float(request.form['size'])
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        age = float(request.form['age'])
        distance = float(request.form['distance'])
        model = pickle.load(open('trained_model.pkl', 'rb'))
        predicted_price = model.predict([[size, bedrooms, bathrooms, age, distance]])
        return render_template('result.html', price=round(predicted_price[0], 2))


if __name__ == '__main__':
    app.run(debug=True)