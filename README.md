# H&M Budget-Constrained Recommendation System

A personalized fashion recommendation pipeline for H&M that retrieves and ranks articles under a user-specified budget. It uses a two-tower model for candidate retrieval via FAISS approximate nearest neighbor search, followed by a SASRec-based reranker that scores candidates using sequential purchase history, and a greedy knapsack algorithm to maximize relevance within the budget.

## Tech Stack

- **Python**, **PyTorch** — model training and inference
- **FAISS** — approximate nearest neighbor retrieval
- **Pandas**, **NumPy**, **scikit-learn** — data preprocessing and feature engineering
- **AWS SageMaker** — managed spot training on GPU instances
- **Amazon S3** — data and model artifact storage
