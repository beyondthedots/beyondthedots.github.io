import subprocess
from datetime import date

print("Running 2012-2014 extraction...")
subprocess.run(["python3", "run_2012-2014.py"])

print("\nRunning 2015+ extraction...")
subprocess.run(["python3", "run_2015-plus.py"])

print("\nDone! CSV saved to outputs/projections.csv")