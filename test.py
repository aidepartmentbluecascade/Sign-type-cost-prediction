import gspread
from google.oauth2.service_account import Credentials

# Define the scope (permissions)
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Path to your JSON key file
creds = Credentials.from_service_account_file("credentials.json", scopes=scope)

# Authorize client
client = gspread.authorize(creds)

# Open your spreadsheet by name or URL
sheet = client.open("sheet-access").sheet1

# Read data
data = sheet.get_all_records()
print(data)

# Write data
sheet.update_cell(2, 2, "Hello from Python!")