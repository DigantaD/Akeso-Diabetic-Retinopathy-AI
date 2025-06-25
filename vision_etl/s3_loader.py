import os
import boto3
import io
import pandas as pd
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

def load_image_from_s3(key):
    """Returns image bytes (PIL-loadable or cv2.imdecode compatible)"""
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return io.BytesIO(obj["Body"].read())

def load_csv_from_s3(key):
    """Returns a Pandas DataFrame from a CSV in S3"""
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def list_objects(prefix):
    """
    Lists all files under a prefix with pagination.
    """
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys

def list_image_keys_from_s3_prefix(prefix, extensions=None):
    """
    Lists keys under a prefix. If `extensions` is None, returns all.
    """
    try:
        if not isinstance(prefix, str):
            raise ValueError(f"[S3 Loader] Prefix must be a string. Got: {type(prefix)} → {prefix}")
        
        all_keys = list_objects(prefix)
        if extensions:
            return [k for k in all_keys if k.lower().endswith(extensions)]
        return all_keys
    except:
        return prefix