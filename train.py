import subprocess
import sys

print("=" * 50)
print("IRIS MACHINE LEARNING PROJECT")
print("=" * 50)

python = sys.executable

print("\nStep 1 : Loading Dataset")
subprocess.run([python, "src/data.py"])

print("\nStep 2 : Feature Profiling")
subprocess.run([python, "src/features.py"])

print("\nStep 3 : Training Model")
subprocess.run([python, "src/model.py"])

print("\nStep 4 : Evaluating Model")
subprocess.run([python, "src/evaluate.py"])

print("\nProject Completed Successfully!")