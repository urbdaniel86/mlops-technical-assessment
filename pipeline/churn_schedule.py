from dagster import schedule

@schedule(
    cron_schedule="0 0 * * 0",  # Every Sunday at midnight, for example
    job=churn_prediction_job,
    execution_timezone="UTC"
)
def weekly_churn_schedule(context):
    return {}
