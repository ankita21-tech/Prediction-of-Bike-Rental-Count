import sys, os
# ensure project root is on path so imports like `import app` work when running from scripts/ folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

# sample form values matching index.html inputs (strings that will be cast)
form = {
    'weather': '1',
    'Seasons': '2',
    'temp': '0.5',
    'hum': '0.5',
    'Month': '7',
    'windspeed': '0.2',
    'yr': '1',
    'holiday': '0'
}

client = app.test_client()
resp = client.post('/predict', data=form)
print('status', resp.status_code)
# print a snippet of the returned HTML so we can see prediction text
print(resp.get_data(as_text=True)[-800:])
