import lmdb
import os
import msgpack
from lz4.frame import compress

def create_lmdb(input_dir, output_dir, map_size=2 * 1024**3):  # 2 GB
    # Create the LMDB environment
    env = lmdb.open(output_dir, map_size=map_size)
    txn = env.begin(write=True)

    try:
        for idx, filename in enumerate(os.listdir(input_dir)):
            if filename.endswith('.gui'):
                file_path = os.path.join(input_dir, filename)
                
                # Read the file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Use the filename (without extension) as the LMDB key
                key = filename.replace('.gui', '').encode('utf-8')
                value = msgpack.dumps(content, use_bin_type=True)
                
                # Compress the content and store in LMDB
                txn.put(key, compress(value))

                # Commit every 500 files to reduce memory pressure
                if (idx + 1) % 500 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    print(f"Processed {idx + 1} files...")
        
        # Final commit
        txn.commit()
        print("LMDB creation complete.")
    
    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        # Close the LMDB environment
        env.close()

# Run the script
create_lmdb(
    input_dir='/Users/michellechang/DiffCap/data/val_texts',
    output_dir='/Users/michellechang/DiffCap/data/lmdb_val_texts',
    map_size=2 * 1024**3  # 2 GB
)
