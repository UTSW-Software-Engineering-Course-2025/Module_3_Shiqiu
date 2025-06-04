"""
mlflow_hello_world.py
A Quick tutorial to logging to mlflow
1. Points MLflow at the tracking server you supply.
2. Sets the experiment to your name.
3. Starts a run.
4. Logs two example parameters.
5. Prints the run ID so you can find it in the UI.
"""

import argparse

import mlflow
from pydantic import AnyUrl


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal MLflow logging example.")
    parser.add_argument(
        "--tracking-uri",
        required=True,
        help="URL of the MLflow tracking server, e.g. http://localhost:5000",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Your name",
    )
    args = parser.parse_args()
    mlflow_uri = AnyUrl(args.tracking_uri)

    # 1 - Tell MLflow where to send data
    mlflow.set_tracking_uri(str(mlflow_uri))
    mlflow.set_experiment(args.name)

    # 2 - Start a run (auto-creates an experiment called 'Default' if none exists)
    with mlflow.start_run(run_name="hello-world") as run:
        # 3 - Log a couple of dummy parameters
        mlflow.log_param("greeting", "hello_mlflow")
        mlflow.log_param("year", 2025)

        # 4 - Give the user something to click on
        print(f"Logged to MLflow run → {run.info.run_id}")


if __name__ == "__main__":
    main()
