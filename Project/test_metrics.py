import urllib.request, json
try:
    r = urllib.request.urlopen('http://localhost:8000/metrics')
    data = json.loads(r.read())
    print("KEYS IN /metrics RESPONSE:", list(data.keys()))
except Exception as e:
    print("ERROR:", e)
