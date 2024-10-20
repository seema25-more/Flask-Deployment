from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import tensorflow as tf

# Create flask app
app = Flask(__name__,template_folder='templates')

# Load the pickle model
model_path = 'E:/Deployment/Model_4.h5'
model = tf.keras.models.load_model(model_path) 


@app.route("/")
def Home():
    return render_template("index.html")
    app.run(debug=True)



#@app.route("/predict", methods=["POST"])
#def predict():
 #   float_features = [float(x) for x in request.form.values()]
 #   features = [np.array(float_features)]
 #   prediction = model.predict(features)

 #    return render_template("index (1).html", prediction_text="The heart disease chances are {}".format(prediction))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        float_features = [float(x) for x in request.form.values()]  # Convert input strings to floats
    except ValueError:
        return render_template("index.html", prediction_text="Invalid input: Please enter numeric values only.")

    features = np.array(float_features).reshape(1, -1)  # Reshape to 2D array (1 sample, num_features)
    

    prediction = model.predict(features)

   
    predicted_class = np.argmax(prediction, axis=1)

    return render_template("index.html", prediction_text="The heart disease chances are {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True, port=5001)