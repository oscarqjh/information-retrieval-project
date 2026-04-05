import pandas as pd
import requests
import json
import math

# Load your CSV
df = pd.read_csv('/Users/potalaboonkitrungpaisarn/Downloads/opinions.csv')

# Replace NaN with empty string to avoid errors
df = df.fillna('')

# Convert to list of dictionaries
docs = df.to_dict(orient='records')

print(f"Total records to index: {len(docs)}")

# Index in batches of 1000
batch_size = 1000
total_batches = math.ceil(len(docs) / batch_size)

for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    batch_num = (i // batch_size) + 1
    
    response = requests.post(
        'http://localhost:8983/solr/opinions/update?commit=true',
        headers={'Content-Type': 'application/json'},
        data=json.dumps(batch)
    )
    
    if response.status_code == 200:
        print(f"Batch {batch_num}/{total_batches} indexed successfully ✅")
    else:
        print(f"Batch {batch_num} FAILED: {response.text}")

print("Done! All records indexed.")
