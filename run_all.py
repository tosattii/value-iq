"""Run the whole pipeline: download -> clean -> train -> dashboard."""
import build_dataset
import fetch_data
import make_dashboard
import train_model

for step in (fetch_data, build_dataset, train_model, make_dashboard):
    print(f"\n=== {step.__name__} ===")
    step.main()
