from pipelines.train_pipeline import run_pipeline
from zenml.client import Client

if __name__ == "__main__":
    link = "/Users/maxkucher/Desktop/weapon_decision/data/vehicles.csv"
    n_trials = 20
    model_name = "RFC"
    print("Pipeline started")
    run_pipeline(link, model_name, n_trials)
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    print("Pipeline is over")


# mlflow ui --backend-store-uri "file:/Users/maxkucher/Library/Application Support/zenml/local_stores/79a9fc9c-2f87-4b7f-8cfb-d44929e4dcfd/mlruns"