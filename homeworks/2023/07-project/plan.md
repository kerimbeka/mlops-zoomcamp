0. Problem description
    a. 2 points: The problem is well described and it's clear what the problem the project solves
1. Experiment tracking and model registry, i.e mlflow locally
    a. 4 points: Both experiment tracking and model registry are used 
2. Workflow orchestration, i.e prefect, prefect deploy locally
    a. 4 points: Fully deployed workflow
3. Model monitoring, i.e evidently, locally
    a. 2 points: Basic model monitoring that calculates and reports metrics
    b. 4 points: Comprehensive model monitoring that sends alerts or runs a conditional workflow (e.g. retraining, generating debugging dashboard, switching to a different model) if the defined metrics threshold is violated
4. Model deployment: Deploy the model in batch, web service or streaming
    a. 2 points: Model is deployed but only locally 
    b. 4 points: The model deployment code is containerized and could be deployed to cloud or special tools for model deployment are used
5. Cloud
    a. 2 points: The project is developed on the cloud OR uses localstack (or similar tool) OR the project is deployed to Kubernetes or similar container management platforms
    b. 4 points: The project is developed on the cloud and IaC tools are used for provisioning the infrastructure
6. Reproducibility:
    a. 4 points: Instructions are clear, it's easy to run the code, and it works. The versions for all the dependencies are specified.
