import requests
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
for d in ("2026-11-24", "2026-11-25", "2026-11-26", "2026-11-27"):
    r = requests.get("https://api.nasdaq.com/api/calendar/economicevents", params={"date": d}, headers=H, timeout=20)
    rows = (r.json().get("data") or {}).get("rows") or []
    print("RAW", d, r.json()["data"].get("asOf"), [(x["gmt"], x["eventName"]) for x in rows if x["country"] == "United States"][:14])
