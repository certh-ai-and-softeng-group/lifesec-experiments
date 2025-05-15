# LIFESEC 2025 Experiments

This repository documents the experimental setup and process used for our submission to the LIFESEC 2025 Conference.

## Steps

### 1. Install SonarQube via Docker

Use the following command to run SonarQube in a Docker container:

```bash
docker run -d --name sonarqube -p 9000:9000 sonarqube
```

Make sure SonarQube is up and running at [http://localhost:9000](http://localhost:9000).

### 2. Download the BigVul Dataset

You can download the dataset here:
```bash
gdown https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V
```

### 3. Convert Functions of the BigVul Dataset to C Files

Transform each function from the dataset into an individual `.c` file and store them in a dedicated folder (to facilitate the analysis through SonarQube):

```bash
python convert_to_c_files.py --input bigvul --output c_functions
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

### 6. Marge the SonarQube results to the BigVul dataset

```python
python csv_maker_from_sast.py
```
### 6. Run the Vulnerability Prediction analysis using LLM 

```python
python sast_llm.py 
```

### 7. Output

The final output is `results.csv`, which contains the results of both SonarQube and Vulnerabilty Prediction analysis. This CSV serves as the basis for our evaluation.

## License

This project is licensed under the MIT License.
