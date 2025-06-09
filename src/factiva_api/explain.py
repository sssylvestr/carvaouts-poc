import requests, json
from pprint import pprint

from time import sleep

from .common import headers, check_response, extractions_url


def create_explain(query_body):
    """
    Create an explain request to check how many documents match a query
    """
    url = f"{extractions_url}/_explain"
    response = requests.post(url, headers=headers, data=json.dumps(query_body))
    return check_response(response)


def check_explain_status(job_id):
    """
    Check the status of an explain job
    """
    url = f"{extractions_url}/{job_id}/_explain"
    response = requests.get(url, headers=headers)
    return check_response(response)


def wait_for_explain_completion(job_id, max_retries=100, sleep_time=10):
    """
    Wait for an explain job to complete
    """
    for i in range(max_retries):
        status = check_explain_status(job_id)
        if not status:
            return None

        current_state = status["data"]["attributes"]["current_state"]
        print(f"Current state: {current_state}")

        if current_state == "JOB_STATE_DONE":
            return status

        if i < max_retries - 1:
            print(f"Waiting {sleep_time} seconds...")
            sleep(sleep_time)

    print("Max retries reached. Job might still be running.")
    return None


def run_explain(input_query, num_samples=5):
    explain_result = create_explain(input_query)

    if explain_result:
        print("\nExplain job created:")
        print(f"Job ID: {explain_result['data']['id']}")

        job_id = explain_result["data"]["id"]
        final_status = wait_for_explain_completion(job_id)

        print(f"\nGetting samples for job ID: {job_id}")
        if final_status and "counts" in final_status["data"]["attributes"]:
            count = final_status["data"]["attributes"]["counts"]
            print(f"\nNumber of documents matching the query: {count}")

            samples_url = f"https://api.dowjones.com/extractions/samples/{job_id}?num_samples={num_samples}"
            samples_response = requests.get(samples_url, headers=headers)
            samples = samples_response.json()
            pprint(samples, compact=True)
            return final_status, samples
