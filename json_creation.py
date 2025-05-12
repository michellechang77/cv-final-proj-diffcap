import os
import json
import msgpack
from lz4.frame import decompress
import lmdb

def create_id2len(lmdb_dir, output_file):
    env = lmdb.open(lmdb_dir, readonly=True)
    id2len = {}

    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            # Decompress and deserialize the data
            text = msgpack.loads(decompress(value), raw=False)
            # Calculate length
            id2len[key.decode('utf-8')] = len(text.split())

    # Save the lengths to JSON
    with open(output_file, 'w') as f:
        json.dump(id2len, f)

    env.close()
    print(f"Created {output_file} successfully.")

# Run the script
create_id2len(
    lmdb_dir='/Users/michellechang/DiffCap/data/lmdb_val_texts',
    output_file='/Users/michellechang/DiffCap/data/lmdb_val_texts/id2len.json'
)
