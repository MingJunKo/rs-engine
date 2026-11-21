import sys, json, requests
sys.path.insert(0, ".")
import extras as X
for wk in ("thisweek", "nextweek"):
    for url in (f"https://nfs.faireconomy.media/ff_calendar_{wk}.json",):
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            print(wk, r.status_code, r.headers.get("content-type"), len(r.content))
            if r.ok:
                d = r.json(); print("  events", len(d), "USD high", sum(1 for x in d if x.get("country") == "USD" and x.get("impact") == "High"))
                print("  sample", json.dumps(d[:2])[:400])
            else:
                print("  body", r.text[:300])
        except Exception as e:
            print(wk, "ERR", e)
print("econ_calendar():", json.dumps(X.econ_calendar(), indent=1)[:3000])
