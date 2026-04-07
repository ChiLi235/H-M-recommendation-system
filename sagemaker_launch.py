"""
sagemaker_launch.py

Launches the full training pipeline as SageMaker Managed Spot Training jobs.

Cost strategy:
  - ml.m5.xlarge  ($0.046/hr spot) for preprocessing
  - ml.g4dn.xlarge ($0.221/hr spot, T4 GPU) for model training
  - Managed Spot + checkpointing → ~70% cheaper than on-demand
  - Estimated total: ~$4.50 per full run  (~10 runs within $50 budget)

S3 layout:
  s3://<BUCKET>/data/           raw CSVs + processed/ parquets
  s3://<BUCKET>/checkpoints/    SageMaker checkpoint output
  s3://<BUCKET>/models/         final model artifacts

Usage:
  python sagemaker_launch.py --bucket my-hm-bucket --stage all
  python sagemaker_launch.py --bucket my-hm-bucket --stage two_tower
  python sagemaker_launch.py --bucket my-hm-bucket --stage reranker
  python sagemaker_launch.py --bucket my-hm-bucket --stage evaluate
"""

import argparse
import os
import time
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

# ── Configuration ──────────────────────────────────────────────────────────────
REGION        = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
ROLE_ARN      = os.environ.get("SAGEMAKER_ROLE_ARN", "")   # required
PYTORCH_VER   = "2.1"
PY_VER        = "py310"

# Spot config — 24h max wait for a spot instance
SPOT_CONFIG = {
    "use_spot_instances": True,
    "max_wait":           86400,   # 24h max wait for spot capacity
    "max_run":            28800,   # 8h max runtime per job
}

# Instance types
CPU_INSTANCE      = "ml.m5.xlarge"    # preprocessing
GPU_INSTANCE      = "ml.g4dn.2xlarge"  # two-tower training (32 GB RAM)
RERANKER_INSTANCE = "ml.g4dn.4xlarge"  # reranker training (64 GB RAM)


def _bucket_uri(bucket: str, prefix: str) -> str:
    return f"s3://{bucket}/{prefix}"


# ══════════════════════════════════════════════════════════════════════════════
# Job: Data preprocessing + feature engineering + dataset building
# ══════════════════════════════════════════════════════════════════════════════

def launch_preprocessing(bucket: str, session: sagemaker.Session, role: str = ROLE_ARN):
    """
    Runs data_preprocessing.py → feature_engineering.py → training_dataset_builder.py
    on a CPU instance. Uploads processed/ to S3.

    Expected S3 input:  s3://<bucket>/data/  (raw CSVs)
    Expected S3 output: s3://<bucket>/data/processed/
    """
    print("Launching preprocessing job …")

    estimator = PyTorch(
        entry_point="train_pipeline.py",
        source_dir=".",
        role=role,
        instance_type=CPU_INSTANCE,
        instance_count=1,
        framework_version=PYTORCH_VER,
        py_version=PY_VER,
        hyperparameters={
            "skip_preprocess": "false",
            "eval_only":       "false",
            # signal to train_pipeline.py: only run steps 1-3, skip model training
            "steps": "preprocess",
        },
        checkpoint_s3_uri=_bucket_uri(bucket, "checkpoints/preprocess"),
        output_path=_bucket_uri(bucket, "data/processed"),
        sagemaker_session=session,
        **SPOT_CONFIG,
    )

    estimator.fit(
        inputs={"data": TrainingInput(_bucket_uri(bucket, "data/"), content_type="text/csv")},
        job_name=f"hm-preprocess-{int(time.time())}",
        wait=True,
    )
    print("Preprocessing job complete.")
    return estimator


# ══════════════════════════════════════════════════════════════════════════════
# Job: Two-Tower training
# ══════════════════════════════════════════════════════════════════════════════

def launch_two_tower(bucket: str, session: sagemaker.Session, role: str = ROLE_ARN):
    """
    Trains Two-Tower model + builds FAISS index.

    S3 input:  s3://<bucket>/data/processed/  (parquet files from preprocessing)
    S3 output: s3://<bucket>/models/two_tower/
    """
    print("Launching Two-Tower training job …")

    estimator = PyTorch(
        entry_point="two_tower_model.py",
        source_dir=".",
        role=role,
        instance_type=GPU_INSTANCE,
        instance_count=1,
        framework_version=PYTORCH_VER,
        py_version=PY_VER,
        hyperparameters={
            "epochs":     15,
            "batch_size": 1024,
            "lr":         3e-4,
        },
        checkpoint_s3_uri=_bucket_uri(bucket, "checkpoints/two_tower"),
        checkpoint_local_path="/opt/ml/checkpoints",
        output_path=_bucket_uri(bucket, "models/two_tower"),
        sagemaker_session=session,
        **SPOT_CONFIG,
    )

    estimator.fit(
        inputs={
            "processed": TrainingInput(
                _bucket_uri(bucket, "data/processed/"),
                content_type="application/x-parquet",
            )
        },
        job_name=f"hm-two-tower-{int(time.time())}",
        wait=True,
    )
    print("Two-Tower training complete.")
    return estimator


# ══════════════════════════════════════════════════════════════════════════════
# Job: SASRec reranker training
# ══════════════════════════════════════════════════════════════════════════════

def launch_reranker(bucket: str, session: sagemaker.Session, role: str = ROLE_ARN):
    """
    Trains SASRec reranker.

    S3 input:  s3://<bucket>/data/processed/  (parquet files)
               s3://<bucket>/models/two_tower/  (FAISS index for hard negative mining — optional)
    S3 output: s3://<bucket>/models/reranker/
    """
    print("Launching SASRec reranker training job …")

    inputs = {
        "processed": TrainingInput(
            _bucket_uri(bucket, "data/processed/"),
        )
    }

    estimator = PyTorch(
        entry_point="reranker_model.py",
        source_dir=".",
        role=role,
        instance_type=RERANKER_INSTANCE,
        instance_count=1,
        framework_version=PYTORCH_VER,
        py_version=PY_VER,
        hyperparameters={
            "epochs":     10,
            "batch_size": 1024,
            "lr":         1e-3,
            "weight_decay": 1e-4,
        },
        checkpoint_s3_uri=_bucket_uri(bucket, "checkpoints/reranker"),
        checkpoint_local_path="/opt/ml/checkpoints",
        output_path=_bucket_uri(bucket, "models/reranker"),
        sagemaker_session=session,
        **SPOT_CONFIG,
    )

    estimator.fit(
        inputs=inputs,
        job_name=f"hm-reranker-{int(time.time())}",
        wait=True,
    )
    print("Reranker training complete.")
    return estimator


# ══════════════════════════════════════════════════════════════════════════════
# Job: Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def launch_evaluation(bucket: str, session: sagemaker.Session, role: str = ROLE_ARN, budget: float = 100.0, skip_tt_only: bool = False):
    """
    Runs inference.py --evaluate on the test split.

    S3 input:  s3://<bucket>/data/processed/
               s3://<bucket>/models/
    S3 output: s3://<bucket>/evaluation/
    """
    print("Launching evaluation job …")

    estimator = PyTorch(
        entry_point="inference.py",
        source_dir=".",
        role=role,
        instance_type=GPU_INSTANCE,
        instance_count=1,
        framework_version=PYTORCH_VER,
        py_version=PY_VER,
        hyperparameters={
            "evaluate":     "",
            "budget":       budget,
            **({"skip_tt_only": ""} if skip_tt_only else {}),
        },
        checkpoint_s3_uri=_bucket_uri(bucket, "checkpoints/evaluate"),
        checkpoint_local_path="/opt/ml/checkpoints",
        output_path=_bucket_uri(bucket, "evaluation"),
        sagemaker_session=session,
        **SPOT_CONFIG,
    )

    estimator.fit(
        inputs={
            "processed": TrainingInput(_bucket_uri(bucket, "data/processed/")),
            "models":    TrainingInput(_bucket_uri(bucket, "model_artifacts/")),
        },
        job_name=f"hm-evaluate-{int(time.time())}",
        wait=True,
    )
    print("Evaluation complete.")
    return estimator


# ══════════════════════════════════════════════════════════════════════════════
# Upload local processed/ to S3
# ══════════════════════════════════════════════════════════════════════════════

def upload_data(bucket: str, local_dir: str = "processed", s3_prefix: str = "data/processed"):
    """Upload local processed/ directory to S3 before launching training jobs."""
    import boto3
    s3 = boto3.client("s3")
    uploaded = 0
    for fname in os.listdir(local_dir):
        local_path = os.path.join(local_dir, fname)
        s3_key = f"{s3_prefix}/{fname}"
        print(f"  Uploading {local_path} → s3://{bucket}/{s3_key}")
        s3.upload_file(local_path, bucket, s3_key)
        uploaded += 1
    print(f"  Uploaded {uploaded} files to s3://{bucket}/{s3_prefix}/")


def upload_raw_data(bucket: str, files: list = None, s3_prefix: str = "data"):
    """Upload raw CSV files to S3."""
    import boto3
    s3 = boto3.client("s3")
    if files is None:
        files = ["articles.csv", "customers.csv",
                 "transactions_train_new.csv", "transactions_train.csv"]
    for f in files:
        if os.path.exists(f):
            s3_key = f"{s3_prefix}/{f}"
            print(f"  Uploading {f} → s3://{bucket}/{s3_key}")
            s3.upload_file(f, bucket, s3_key)
        else:
            print(f"  WARNING: {f} not found, skipping")


# ══════════════════════════════════════════════════════════════════════════════
# Cost estimate
# ══════════════════════════════════════════════════════════════════════════════

def print_cost_estimate():
    print("""
Cost estimate (Managed Spot, us-east-1):
  ─────────────────────────────────────────────────────────
  Step               Instance          Rate      Est. Time  Est. Cost
  Preprocessing      ml.m5.xlarge      $0.046/hr   2h       $0.09
  Two-Tower train    ml.g4dn.xlarge    $0.221/hr   5h       $1.11
  Reranker train     ml.g4dn.xlarge    $0.221/hr   8h       $1.77
  Evaluation         ml.g4dn.xlarge    $0.221/hr   2h       $0.44
  S3 storage 50GB    Standard          $0.023/GB   1mo      $1.15
  ─────────────────────────────────────────────────────────
  TOTAL PER RUN                                            ~$4.56

  $50 budget ≈ 10 full training runs (hyperparameter tuning / ablations)
""")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Launch SageMaker training jobs")
    parser.add_argument("--bucket",         required=True,          help="S3 bucket name")
    parser.add_argument("--role",           default=ROLE_ARN,       help="SageMaker IAM role ARN")
    parser.add_argument("--stage",          default="all",
                        choices=["all", "upload", "preprocess", "two_tower",
                                 "reranker", "evaluate", "cost"],
                        help="Which stage to run")
    parser.add_argument("--budget",         type=float, default=100.0)
    parser.add_argument("--skip_tt_only",   action="store_true",    help="Skip two-tower-only evaluation")
    parser.add_argument("--upload_raw",     action="store_true",    help="Upload raw CSVs to S3 first")
    args = parser.parse_args()

    if args.stage == "cost":
        print_cost_estimate()
        return

    if not args.role:
        print("ERROR: set --role or export SAGEMAKER_ROLE_ARN=...")
        return

    role = args.role
    session = sagemaker.Session(boto3.session.Session(region_name=REGION))

    if args.upload_raw:
        upload_raw_data(args.bucket)

    if args.stage in ("all", "upload"):
        print("Uploading processed/ to S3 …")
        upload_data(args.bucket)

    if args.stage in ("all", "preprocess"):
        launch_preprocessing(args.bucket, session, role)

    if args.stage in ("all", "two_tower"):
        launch_two_tower(args.bucket, session, role)

    if args.stage in ("all", "reranker"):
        launch_reranker(args.bucket, session, role)

    if args.stage in ("all", "evaluate"):
        launch_evaluation(args.bucket, session, role, budget=args.budget, skip_tt_only=args.skip_tt_only)

    print("\nAll requested jobs complete.")


if __name__ == "__main__":
    main()
