# LIFESEC 2025 Experiments

This repository documents the experimental setup and process used for our submission to the LIFESEC 2025 Conference. The goal of the experiment is to assess vulnerabilities in C functions from the BigVul dataset using SonarQube static analysis.

## Steps

### 1. Install SonarQube via Docker

Use the following command to run SonarQube in a Docker container:

```bash
docker run -d --name sonarqube -p 9000:9000 sonarqube
```

Make sure SonarQube is up and running at [http://localhost:9000](http://localhost:9000).

### 2. Download the BigVul Dataset

You can download the dataset from Hugging Face:

[https://huggingface.co/datasets/bstee615/bigvul](https://huggingface.co/datasets/bstee615/bigvul)

Use the `datasets` library to download it in Python:

```python
from datasets import load_dataset

dataset = load_dataset("bstee615/bigvul")
```

### 3. Convert Functions of the BigVul Dataset to C Files

Transform each function from the dataset into an individual `.c` file and store them in a dedicated folder:

```bash
python scripts/convert_to_c_files.py --input bigvul --output c_functions
```

> Ensure the script extracts and writes each function as a valid C file.

### 4. Execute SonarQube Analysis

Run SonarQube analysis on the folder containing the C files using the SonarScanner:

```bash
sonar-scanner   -Dsonar.projectKey=bigvul-analysis   -Dsonar.sources=./c_functions   -Dsonar.host.url=http://localhost:9000   -Dsonar.login=YOUR_SONAR_TOKEN
```

> Replace `YOUR_SONAR_TOKEN` with your actual token from SonarQube.

### 5. Fetch Issues

Use the provided `fetch_issues.py` script to extract the results from the SonarQube instance.
This will generate a CSV file containing all identified issues from the SonarQube analysis.

### 6. Output

The final output is `results.csv`, which contains the SonarQube-reported issues for each function analyzed. This CSV serves as the primary dataset for further analysis or evaluation.

## License

This project is licensed under the MIT License.
