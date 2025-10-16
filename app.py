from flask import Flask, request, render_template
from pipeline import initialize_predictor

# Initialize Flask app
app = Flask(__name__)

# Initialize the ML predictor
print("Initializing machine learning models...")
predictor = initialize_predictor()
print("Models trained successfully!")

@app.route("/", methods=["GET", "POST"])
def predict_cost():
    result = None
    if request.method == "POST":
        try:
            sign_type = request.form['sign_type']
            width = float(request.form['width'])
            height = float(request.form['height'])
            
            # Get prediction from the ML pipeline
            result = predictor.predict_costs(sign_type, width, height)

            actual = predictor.find_actual(sign_type, width, height)
            if actual:
                result.update({
                    'actual_production_cost': actual['production_cost'],
                    'actual_shipping_cost': actual['shipping_cost'],
                    'actual_total_cost': actual['total_cost'],
                    'exact_match': True
                })
            else:
                result['exact_match'] = False
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            result = {
                'error': 'An error occurred during prediction. Please check your inputs.'
            }
    
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
