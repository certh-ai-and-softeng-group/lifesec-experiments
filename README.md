# LIFESEC 2025 Experiments

This repository documents the experimental setup and process used for our submission to the SmartComp/LIFESEC 2025 Conference:

### Replication Package of "AI-Enhanced Static Analysis: Reducing False Alarms Using Large Language Models"

To replicate the analysis and reproduce the results follow the steps below:

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

This script return the evaluation results.

Download and extract the fine-tuned VP model:

https://mega.nz/file/IGU2GZbB#Wa3-KsP0AB0CozZbWAAwmzl-hk5cLBael2ILRR_1EJE

and then run the script:

```python
python sast_llm.py 
```
More information about the extraction of the model can be found here:

https://github.com/iliaskaloup/vulGpt 

### 7. Output

The final output is `filtered_bigvul_results.csv`, which contains the results of both SonarQube and Vulnerabilty Prediction analysis.

## Acknowledgements

Special thanks to the paper entitled "A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries" for providing the Big-Vul dataset, which was utilized as the basis for our analysis. If you use Big-Vul, please cite:

~~~
J. Fan, Y. Li, S. Wang and T. N. Nguyen, "A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries," 2020 IEEE/ACM 17th International Conference on Mining Software Repositories (MSR), Seoul, Korea, Republic of, 2020, pp. 508-512, doi: 10.1145/3379597.3387501
~~~


## License

This project is licensed under the MIT License.

## Citation

Apostolidis, G., Kalouptsoglou, I., Siavvas, M., Kehagias, D., & Tzovaras, D. AI-Enhanced Static Analysis: Reducing False Alarms Using Large Language Models. In 11th IEEE International Conference on Smart Computing, Ireland, 2025
