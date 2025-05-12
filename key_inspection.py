import lmdb

def inspect_keys(db_path):
    env = lmdb.open(db_path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            print(key.decode("utf-8"))
    env.close()

inspect_keys("/Users/michellechang/DiffCap/data/train_texts")
