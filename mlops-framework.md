# MLOps Framework for Collaborative Data Science Development and Deployment

## Overview

The primary objective is to develop data science projects collaboratively and deploy models rapidly, embracing the principles of MLOps. This framework addresses the entire stack from development to deployment, ensuring seamless collaboration, environment isolation, and efficient data handling.

## Key Components

1. **Collaborative Development**
2. **Environment Isolation**
3. **Data Ingestion Methods**
4. **Compute Resources**
5. **Storage Solutions**
6. **Platform-Agnostic Codebase**
7. **Pipelines for Data Processing and Deployment**
8. **Platform Constraints**

---

## 1. Collaborative Development

- **Version Control with GitHub**:
  - Each project resides in its own repository.
  - Use short-lived issues and branches for focused development.
  - Collaborate via pull requests, merging into `dev`, then promoting to `staging` and `main` after successful testing.
  - **Branch Policies**:
    - `main` and `staging` branches must always be production-ready.
    - Multiple contributors can work concurrently on different issues in separate branches.

---

## 2. Environment Isolation

- **Separate Environments**:
  - **Development (`dev`)**
  - **Staging**
  - **Production (`main`)**
- **Data Isolation**:
  - Each environment uses separate, isolated data buckets to prevent cross-contamination.
  - Processes in one environment do not interfere with those in others.
- **Environment Detection**:
  - Code identifies the current environment via branch names or environment variables (useful for Docker containers).

---

## 3. Data Ingestion Methods

1. **BigQuery SQL Queries**:
   - Execute SQL queries directly on BigQuery.
   - Load results into DataFrames within the Python runtime.

2. **Dataproc**:
   - Use Dataproc for data processing tasks.
   - Store output as `.parquet` files in Google Cloud Storage buckets.

3. **Reading Static Files**:
   - Utilize the Polars library to read `.parquet` files from buckets.

---

## 4. Compute Resources

- **Virtual Machines (GCP Vertex AI Workbench Instances)**:
  - Primarily for model training and development.
  - Limited CPU and RAM; heavy data processing offloaded to BigQuery or Dataproc.
  - Handle large datasets with Polars and LazyFrames to process data larger than memory.

- **Kubeflow Pipelines**:
  - For stable, scheduled production workloads.
  - Perform model training and data processing in a scalable manner.

- **Airflow DAGs**:
  - Used within the proprietary Element platform.
  - Schedule inference runs for trained models.

---

## 5. Storage Solutions

- **Google Cloud Storage Buckets**:
  - Store all data in `.parquet` format for consistency and efficiency.
  - Temporary objects can be stored as `.pkl` files (not recommended for long-term storage).

- **MLflow Server**:
  - Manage long-term storage of models with dependencies and metadata.
  - Facilitate model versioning and lifecycle management.
  - Decouple training from inference; inference processes retrieve the latest models from the registry.

---

## 6. Platform-Agnostic Codebase

- **Reusable Code Principles**:
  - Design code to be platform-agnostic, executable across different environments (local Python runtime, Kubeflow Pipelines, Airflow).

- **Shared Library Modules**:

  - **Databases Module**:
    - Load and store objects to buckets via serialization (parquet or pickle) using Polars.
    - Execute SQL queries through BigQuery.
    - Handle various I/O-related procedures.

  - **Credentials Module**:
    - Manage authentication for Vertex AI and the Element platform.
    - Access GCP Secrets to handle environment-specific dependencies.

  - **Pipeline Module**:
    - Provide convenience functions for local pipeline execution.
    - Transform pipelines for deployment to Kubeflow with the same input configurations.

  - **Models Module**:
    - Implement an `MlLogger` class for interacting with MLflow.
    - Store models and artifacts, retrieve the latest models, and manage model metadata and tags.

---

## 7. Pipelines for Data Processing and Deployment

- **Functional Programming Approach**:
  - Write business logic (data transformations, computations) as pure functions.
  - Functions accept DataFrames as inputs and return DataFrames or other objects.
  - Minimize I/O operations within these functions to enhance testability and reusability.

- **Declarative Pipeline Definition**:
  - Define pipelines as a list of steps, where each step is a dictionary containing:
    - `inputs`: List of input names (strings).
    - `outputs`: List of output names (strings).
    - `function`: Callable function to execute.
    - `kwargs`: Additional keyword arguments for the function.

- **Pipeline Execution**:
  - A dedicated function executes each step sequentially, handling all I/O operations.
  - Outputs from functions are automatically stored in buckets.
  - Subsequent functions declare required inputs by referencing the output names.

- **Local and Cloud Deployment**:
  - The same pipeline definition can be run locally or transformed into a Kubeflow pipeline.
  - Scripts are available to manage credentials and environment specifics seamlessly.

---

## 8. Platform Constraints

- **No Local Disk Storage**:
  - All data communication and storage must occur through Google Cloud Storage buckets.
  - Local disk storage is not permitted due to platform limitations.

- **Polars Library Utilization**:
  - Chosen for efficient I/O operations with GCP buckets.
  - Capable of processing datasets larger than available memory via lazy evaluation and direct streaming.

---

## Conclusion

This MLOps framework facilitates collaborative development and rapid deployment of data science projects. By leveraging GitHub for collaboration, enforcing environment isolation, and utilizing GCP services like BigQuery, Dataproc, and MLflow, the team can maintain production stability while efficiently handling data processing and model lifecycle management. The platform-agnostic codebase and declarative pipeline definitions ensure that the code is reusable and maintainable across different environments, aligning with best practices in modern MLOps.

---

## Next Steps for the Team

- **Familiarize with Shared Library Modules**:
  - Review the modules (`Databases`, `Credentials`, `Pipeline`, `Models`) to understand their functionalities and usage.

- **Adopt the Pipeline Definition Standard**:
  - Begin defining new pipelines using the declarative approach to ensure consistency and ease of deployment.

- **Enforce Environment Policies**:
  - Ensure all team members are aware of the environment isolation policies and adhere to them during development.

- **Leverage MLflow for Model Management**:
  - Start integrating MLflow into the model training and deployment processes for better version control and metadata tracking.

- **Optimize Data Handling with Polars**:
  - Encourage the use of the Polars library for data processing tasks to take advantage of its performance benefits.

---

By adopting this structured approach, the team can enhance collaboration, improve code quality, and accelerate the deployment of data science projects, all while maintaining robust production systems.
