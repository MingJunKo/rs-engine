import sys, json, requests, datetime as dt
sys.path.insert(0, ".")
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
d = dt.date.today() + dt.timedelta(days=3)
r = requests.get("https://api.nasdaq.com/api/calendar/economicevents", params={"date": d.isoformat()}, headers=H, timeout=20)
print(d, r.status_code, r.text[:1500])
