import requests
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
for d in ("2026-11-22","2026-11-23","2026-11-24","2026-11-25","2026-11-26","2026-11-27","2026-11-28","2026-11-30"):
    r = requests.get("https://api.nasdaq.com/api/calendar/economicevents", params={"date": d}, headers=H, timeout=20)
    j = r.json(); rows = (j.get("data") or {}).get("rows") or []
    print("RAW", d, (j.get("data") or {}).get("asOf"), len(rows), [(x["gmt"], x["eventName"]) for x in rows if x["country"] == "United States"][:10])
