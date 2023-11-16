import requests

url = 'http://localhost:5000/api'

r = requests.post(url, json={'country': 'United States of America', 'ed_level': 'Bachelor’s degree', 'years_code': 5})

print(r.json())