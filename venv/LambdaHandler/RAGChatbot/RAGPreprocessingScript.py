#This script preprocesses text and PDF files from an S3 bucket, chunks the text, cleans it, and generates embeddings using AWS Bedrock.
#It saves the processed data as a JSON file in the same S3 bucket.
#This script is however used in a Lambda function that is triggered by an S3 event when new files are uploaded to /deleted from the bucket.

import boto3
import json
import string
import re
import requests
from bs4 import BeautifulSoup
import time
import botocore
import urllib3

# Disable SSL warnings for internal pages with self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

s3 = boto3.client("s3")
bucket = "test-bucket-chatbot-321"
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

MAX_CHUNK_LENGTH = 8000

def get_embedding(text, max_retries=5, delay=1):
    for attempt in range(max_retries):
        try:
            body = json.dumps({
                "texts": [text],
                "input_type": "search_document"
            })
            response = bedrock.invoke_model(
                modelId="cohere.embed-english-v3",
                body=body
            )
            return json.loads(response["body"].read())["embeddings"][0]
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == "ThrottlingException":
                print(f"Throttled. Waiting {delay} seconds before retrying...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"Embedding failed: {e}")
                break
        except Exception as e:
            print(f"Embedding failed: {e}")
            break
    print("Max retries exceeded for embedding.")
    return None

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def clean_chunk(chunk):
    chunk = ''.join(c for c in chunk if c in string.printable)
    chunk = re.sub(r'\s+', ' ', chunk)
    return chunk.strip()

def is_valid_chunk(chunk):
    if not chunk or not chunk.strip():
        return False
    try:
        chunk.encode('utf-8')
    except UnicodeEncodeError:
        return False
    return True

def scrape_webpage(url):
    try:
        # For internal/trusted pages, ignore SSL verification
        resp = requests.get(url, timeout=10, verify=False)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        return text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return ""

kb_index = []

objects = s3.list_objects_v2(Bucket=bucket)
for obj in objects.get("Contents", []):
    key = obj["Key"]
    if key == "kb_index.json":
        continue
    if key.strip().lower().endswith(".txt"):
        print(f"Processing text file: {key}")
        file_obj = s3.get_object(Bucket=bucket, Key=key)
        text = file_obj["Body"].read().decode("utf-8")
    else:
        continue

    print(f"Extracted text from {key}: {repr(text[:200])}")
    if text.strip():
        for chunk in chunk_text(text):
            chunk = clean_chunk(chunk)
            if not is_valid_chunk(chunk):
                continue
            if len(chunk) > MAX_CHUNK_LENGTH:
                chunk = chunk[:MAX_CHUNK_LENGTH]
            try:
                embedding = get_embedding(chunk)
                if embedding is not None:
                    kb_index.append({
                        "chunk": chunk,
                        "embedding": embedding,
                        "source": key
                    })
                time.sleep(0.2)
            except Exception as e:
                print(f"Embedding failed for chunk from {key}: {e}")

# --- Web scraping section ---
urls = [
    "https://en.wikipedia.org/wiki/Amazon_Web_Services",
    "https://w.amazon.com/bin/view/Transportation/Passport/Passport_QA_Process/"
    # Add more URLs as needed
]

for url in urls:
    print(f"Scraping web page: {url}")
    text = scrape_webpage(url)
    print(f"Extracted text from {url}: {repr(text[:200])}")
    if text.strip():
        for chunk in chunk_text(text):
            chunk = clean_chunk(chunk)
            if not is_valid_chunk(chunk):
                continue
            if len(chunk) > MAX_CHUNK_LENGTH:
                chunk = chunk[:MAX_CHUNK_LENGTH]
            try:
                embedding = get_embedding(chunk)
                if embedding is not None:
                    kb_index.append({
                        "chunk": chunk,
                        "embedding": embedding,
                        "source": url
                    })
                time.sleep(0.2)
            except Exception as e:
                print(f"Embedding failed for chunk from {url}: {e}")

if kb_index:
    try:
        print(f"Saving {len(kb_index)} chunks to S3...")
        s3.put_object(
            Bucket=bucket,
            Key="kb_index.json",
            Body=json.dumps(kb_index).encode("utf-8")
        )
        print("kb_index.json successfully written to S3.")
    except Exception as e:
        print(f"Failed to write kb_index.json: {e}")
else:
    print("No valid chunks found. Not updating kb_index.json.")

print(f"Final total chunks: {len(kb_index)}")