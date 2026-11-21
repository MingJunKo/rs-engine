import sys
sys.path.insert(0, ".")
import extras as X
ev = X.econ_calendar()
print("ECON", len(ev)); [print("ECON", e["when"], "|", e["title"], "|", e["forecast"], "|", e["previous"]) for e in ev]
