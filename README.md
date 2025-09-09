The goal of this project is to develop, track, and deploy an end-to-end Lung Cancer Identifier using CT scan images.
A fine-tuned ResNet18 CNN is used for accurate feature extraction and binary classification (Cancer vs. Normal).
The workflow integrates MLflow for experiment tracking, DVC for dataset and pipeline management, and Docker + GitHub Actions for automated CI/CD deployment to cloud platforms.
![lung cancer identifier Pipeline] (assets/pipeline.png)