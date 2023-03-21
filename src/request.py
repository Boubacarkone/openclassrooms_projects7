import requests

url = 'http://127.0.0.1:1080/predict'  # localhost and the defined port + endpoint
SK_ID_CURR = 100001
data = {'SK_ID_CURR': SK_ID_CURR}

response = requests.post(url, data=data)
print(response.json())
