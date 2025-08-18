import boto3
import json

s3 = boto3.client("s3")
bucket = "test-bucket-chatbot-321"
key = "kb_index.json"

# Download and load the index
obj = s3.get_object(Bucket=bucket, Key=key)
kb_index = json.loads(obj["Body"].read().decode("utf-8"))

print(f"Total chunks: {len(kb_index)}")

# Check structure and embedding consistency
for i, entry in enumerate(kb_index):
    if "chunk" not in entry or "embedding" not in entry:
        print(f"Entry {i} missing 'chunk' or 'embedding'")
    if not isinstance(entry["chunk"], str):
        print(f"Entry {i} 'chunk' is not a string")
    if not isinstance(entry["embedding"], list):
        print(f"Entry {i} 'embedding' is not a list")
    if not all(isinstance(x, (float, int)) for x in entry["embedding"]):
        print(f"Entry {i} 'embedding' contains non-numeric values")
    if i == 0:
        emb_len = len(entry["embedding"])
    else:
        if len(entry["embedding"]) != emb_len:
            print(f"Entry {i} embedding length {len(entry['embedding'])} does not match first ({emb_len})")