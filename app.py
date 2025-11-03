from flask import Flask, request, render_template, jsonify
from pipeline import initialize_predictor
from google.oauth2.service_account import Credentials
import gspread
import os, json, base64
from dotenv import load_dotenv
from datetime import datetime
import pytz
app = Flask(__name__)

print("Initializing machine learning models...")
predictor = initialize_predictor()
print("Models trained successfully!")


load_dotenv()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds_raw = os.getenv("GOOGLE_CREDENTIALS")

if not creds_raw:
    raise ValueError("❌ GOOGLE_CREDENTIALS not found in .env file")

try:
    creds_info = json.loads(creds_raw)
except json.JSONDecodeError as e:
    raise ValueError(f"❌ Failed to parse GOOGLE_CREDENTIALS JSON: {e}")

creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")

creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
client = gspread.authorize(creds)

print("✅ Google Sheets connected successfully!")


try:
    sheet_access = client.open("sheet-access")
except gspread.SpreadsheetNotFound:
    sheet_access = client.create("sheet-access")
    print("🆕 Created new sheet: sheet-access")

# Ensure worksheets exist
def ensure_worksheet(sheet_title):
    try:
        ws = sheet_access.worksheet(sheet_title)
    except gspread.WorksheetNotFound:
        ws = sheet_access.add_worksheet(title=sheet_title, rows="1000", cols="10")
        print(f"🆕 Created worksheet: {sheet_title}")

    # Add header row if sheet is empty
    if not ws.get_all_values():
        headers = [
            "timestamp",
            "sign type",
            "height",
            "width",
            "Predicted - shipping Cost",
            "Predicted - Production Cost"
        ]
        ws.append_row(headers)
        print(f"✅ Added header row to {sheet_title}")
    return ws

sheet1 = ensure_worksheet("Acceptable")
sheet2 = ensure_worksheet("UnAcceptable")


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
            result = {'error': 'An error occurred during prediction. Please check your inputs.'}

    return render_template('index.html', result=result)


@app.route("/feedback", methods=["POST"])
def save_feedback():
    """Receives feedback via AJAX and saves to Google Sheets"""
    try:
        data = request.get_json()
        print("📩 Feedback received:", data)

        sign_type = data.get("sign_type")
        width = data.get("width")
        height = data.get("height")
        production_cost = data.get("production_cost")
        shipping_cost = data.get("shipping_cost")
        feedback = data.get("feedback")

        pakistan_tz = pytz.timezone('Asia/Karachi')
        timestamp = datetime.now(pakistan_tz).strftime("%Y-%m-%d %H:%M:%S")
    
        row = [
            timestamp,
            sign_type,
            height,
            width,
            shipping_cost,
            production_cost
        ]

        target_sheet = sheet1 if feedback.lower() == "acceptable" else sheet2
        target_sheet.append_row(row)
        print(f"✅ Data appended to {feedback} sheet")

        return jsonify({"status": "success"}), 200

    except Exception as e:
        print("❌ Error saving feedback:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
