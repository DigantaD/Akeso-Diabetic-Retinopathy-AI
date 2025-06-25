import os
import boto3
from dotenv import load_dotenv
import botocore

load_dotenv()

# AWS credentials (auto loaded from .env)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = "akeso-eyecare"

# S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
    config=botocore.client.Config(signature_version='s3v4'),
    use_ssl=False
)

def download_model_from_s3(bucket: str, s3_key: str, local_path: str):
    if os.path.exists(local_path):
        print(f"[✓] Model already cached: {local_path}")
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"[↓] Downloading {s3_key} from S3 to {local_path}")
    s3.download_file(bucket, s3_key, local_path)
    return local_path