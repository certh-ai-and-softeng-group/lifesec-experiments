import requests
import json

# Configuration
SONARQUBE_HOST = "http://localhost:9000"
PROJECT_KEY = "bigvul-analysis "
SONARQUBE_TOKEN = ""  # Replace with actual token
PAGE_SIZE = 500  


HEADERS = {
    "Authorization": f"Bearer {SONARQUBE_TOKEN}"
}


SEVERITIES = ["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"]

def fetch_cpp_issues_batch(project_key, severity):
    """ Fetch issues in batches of 10,000 using severities to bypass API limits """
    page_number = 1
    all_issues = []

    while True:
        url = (f"{SONARQUBE_HOST}/api/issues/search?"
               f"componentKeys={project_key}&tags=cppcheck&severities={severity}&ps={PAGE_SIZE}&p={page_number}")

        response = requests.get(url, headers=HEADERS)

        if response.status_code != 200:
            print(f"Error fetching {severity} issues: {response.status_code}")
            break

        json_data = response.json()
        issues = json_data.get("issues", [])

        if not issues:
            break  # Stop if no more issues found

        all_issues.extend(issues)
        print(f"Fetched {len(issues)} {severity} issues (Page {page_number})")

        # Stop if we reach 10,000 issues (API limit per batch)
        if len(all_issues) >= 10000:
            break

        page_number += 1

    return all_issues


def save_issues_to_file(issues, filename="bigvul_cpp_issues.json"):
    """ Save issues to a JSON file """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(issues, file, indent=4)
    print(f"Saved {len(issues)} issues to {filename}")


if __name__ == "__main__":
    all_issues = []

    # Fetch issues by severity to bypass the 10K limit
    for severity in SEVERITIES:
        issues = fetch_cpp_issues_batch(PROJECT_KEY, severity)
        all_issues.extend(issues)

    save_issues_to_file(all_issues)
