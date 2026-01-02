from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, predictpipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")  # form page

    # POST
    data = CustomData(
        gender=request.form.get("gender"),
        race_ethnicity=request.form.get("ethnicity"),
        parental_level_of_education=request.form.get("parental_level_of_education"),
        lunch=request.form.get("lunch"),
        test_preparation_course=request.form.get("test_preparation_course"),
        reading_score=request.form.get("reading_score"),
        writing_score=request.form.get("writing_score"),
    )

    df = data.get_data_as_dataframe()
    prediction = predictpipeline()
    results = prediction.load_models(df)

    return render_template("home.html", results=results[0])


if __name__ == "__main__":
    # This will print full info like:
    # * Serving Flask app 'app'
    # * Debug mode: on
    # * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    app.run()  # or set host/port explicitly
    # app.run(host="0.0.0.0", port=5000, debug=True)
