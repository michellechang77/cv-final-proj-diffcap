import os
import lmdb
import json
from tqdm import tqdm
from lz4.frame import compress

def generate_lmdb(data_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the LMDB environment
    env = lmdb.open(output_dir, map_size=4 * 1024**4, writemap=True)
    txn = env.begin(write=True)

    # Prepare id2len.json
    id2len = {}
    total_files = 0
    errors = 0

    # Iterate over all .gui files
    for filename in tqdm(os.listdir(data_dir), desc=f"Processing {data_dir}"):
        try:
            if filename.endswith(".gui"):
                file_path = os.path.join(data_dir, filename)
                
                # Read and compress the content
                with open(file_path, "r") as f:
                    content = f.read().strip()
                    gui_id = filename.replace(".gui", "")
                    id2len[gui_id] = len(content.split())
                    
                    # Compress and store in LMDB
                    compressed_data = compress(content.encode("utf-8"))
                    txn.put(gui_id.encode("utf-8"), compressed_data)
                    total_files += 1
        except Exception as e:
            errors += 1
            print(f"⚠️ Error processing {filename}: {e}")

    # Commit the transaction
    txn.commit()
    env.close()

    # Save id2len.json
    id2len_path = os.path.join(output_dir, "id2len.json")
    with open(id2len_path, "w") as f:
        json.dump(id2len, f, indent=4)

    print(f"✅ LMDB database generated successfully for {data_dir}.")
    print(f"✅ id2len.json generated at: {id2len_path}")
    print(f"📊 Processed {total_files} files with {errors} errors.")

if __name__ == "__main__":
    # Generate LMDB for both train and val directories
    directories = {
        "train": "./data/train_texts",
        "val": "./data/val_texts"
    }

    for name, data_dir in directories.items():
        output_dir = os.path.join(data_dir)
        generate_lmdb(data_dir, output_dir)
