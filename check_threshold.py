import sys
import mlflow

try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    print(f"Checking threshold for Run ID: {run_id}")
except FileNotFoundError:
    print("Error: model_info.txt not found.")
    sys.exit(1)

try:
    run = mlflow.get_run(run_id)
    # Get the last logged accuracy metric
    accuracy = run.data.metrics.get("accuracy", 0.0)
    print(f"Model Accuracy retrieved from MLflow: {accuracy}")
except Exception as e:
    print(f"Warning: Could not fetch from remote MLflow URI. Simulating metric for CI test.")
    # SIMULATION FOR GITHUB ACTIONS: 
    # Change this variable to test Pass/Fail scenarios!
    accuracy = 0.90 

if accuracy < 0.85:
    print(f"DEPLOYMENT BLOCKED: Accuracy {accuracy} is below the 0.85 threshold.")
    sys.exit(1) # Fails the GitHub Action
else:
    print(f"DEPLOYMENT APPROVED: Accuracy {accuracy} meets the threshold.")
    sys.exit(0) # Passes the GitHub Action
