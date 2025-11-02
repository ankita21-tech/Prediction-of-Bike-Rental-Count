import sys, os
# ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

client = app.test_client()
resp = client.post('/filter', data={'dataset':'hour','weekday':'Mon','hour':'9'})
print('status', resp.status_code)
txt = resp.get_data(as_text=True)
print('contains filtered header?', 'Filtered dataset' in txt)
if 'Filtered dataset' in txt:
	start = txt.find('Filtered dataset')
	print(txt[start:start+300])
