Assignment 3: End-to-End Hugging Face Model Training & Docker Deployment
Objective
In this assignment, you will build a complete machine learning workflow starting from a notebook, converting it into production-ready scripts, training and evaluating a model using Hugging Face tools, containerizing the workflow using Docker, and publishing artifacts to GitHub and your Hugging Face profile.
Prerequisites
Python
Basic Machine Learning concepts
Docker fundamentals
Git and GitHub usage
Basic understanding of Hugging Face ecosystem
Assignment Tasks
Task 1: Download Shared Notebook
Link: https://colab.research.google.com/drive/1-DhcPi4j3VBVFt9K39e895aIkXyNLJwS?usp=sharing
Download the ML_DL_Ops_Ass_3-Fine-Tuning-Classification.ipynb file provided by the instructor.
Place it inside your project working directory.
Task 2: Create Your Environment Using Docker
Write a Dockerfile using an appropriate base image.
Install required dependencies.
Set up working directory.
Build and run your container.
Verify Python and required libraries are working inside the container.
Task 3: Convert Notebook to Python Scripts
Convert the downloaded notebook to .py script(s).
Clean unnecessary notebook artifacts.
Organize scripts into modules (train, eval, data, utils).
Task 4: Load Models from Hugging Face
Select a suitable pre-trained model (can be the same as per ipynb file).
Load tokenizer and model.
Document why you selected that model.
Task 5: Train Model Using Trainer API
Prepare dataset.
Configure training arguments.
Train using Trainer API.
Log training metrics.
Task 6: Evaluate Model
Run evaluation on validation/test data.
Record Accuracy / F1 / Loss (as applicable).
Save evaluation results.
Task 7: Save Model to Your Hugging Face Profile
Create or use your Hugging Face account.
Push model, tokenizer, and training config.
Ensure model is publicly accessible.
Task 8: Re-evaluate Model from Your HuggingFace Repo
Load model from your uploaded repository.
Run evaluation again.
Compare metrics with local model evaluation.
Task 9: Create Final Docker Image (Evaluation Only)
Create production Docker image that pulls model from your Hugging Face profile.
Run evaluation automatically inside container on startup.
Task 10: Push Everything to GitHub
Push source code, Dockerfile, requirements file, and README.
Include evaluation results and model link.
Submission Requirements
GitHub Repository Link
Hugging Face Model Link
Docker Image Build Instructions
Short Report (explaining model selection, training summary, evaluation comparison, challenges)
