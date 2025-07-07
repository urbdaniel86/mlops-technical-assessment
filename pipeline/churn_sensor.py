from dagster import sensor, RunRequest
import os

watched_folder = "../data/incoming/" # Example of a folder that we want to monitor

@sensor(job=churn_prediction_job)
def file_drop_sensor(context):
    files = os.listdir(watched_folder)
    if files:
        filename = files[0]
        yield RunRequest(
            run_key=filename,
            run_config={
                "ops": {
                    "load_test_data": {"config": {"filename": filename}} 
                }
            } # We would include the filename in the config and pass it to the pipeline
        )
        # The pipeline would include an op to move the file to the ../data/processed folder after the pipeline is run