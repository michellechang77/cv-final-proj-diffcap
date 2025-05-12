import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict
import json
import msgpack
import msgpack_numpy
import numpy as np
from os.path import exists
from PIL import Image
from lz4.frame import compress, decompress
msgpack_numpy.patch()
import lmdb
import os 


class TxtLmdb(object):
    def __init__(self, db_dir, readonly=True):
        self.readonly = readonly
        self.env = None  # Initialize to None
        if readonly:
            self.env = lmdb.open(db_dir, readonly=True, create=False, readahead=True)
            self.txn = self.env.begin(buffers=True)
            self.write_cnt = None
        else:
            self.env = lmdb.open(db_dir, readonly=False, create=True, map_size=4 * 1024**4)
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0

    def __del__(self):
        if self.env is not None:
            if self.write_cnt is not None:
                self.txn.commit()
            self.env.close()

    def __getitem__(self, key):
        data = self.txn.get(key.encode('utf-8'))
        return msgpack.loads(decompress(data), raw=False) if data else None

    def __setitem__(self, key, value):
        if self.readonly:
            raise ValueError('Cannot write to read-only TxtLmdb')
        self.txn.put(key.encode('utf-8'), compress(msgpack.dumps(value, use_bin_type=True)))
        self.write_cnt += 1
        if self.write_cnt % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

class TxtTokLmdb(object):
    def __init__(self, db_dir, max_txt_len=64, use_bert_tokenizer=False):
        self.fix_size = max_txt_len
        max_txt_len = max_txt_len - 1  # sep token
        self._txt2img = None  # Initialize the txt2img attribute
        
        # Initialize the length dictionary
        if max_txt_len == -1:
            self.id2len = json.load(open(f'{db_dir}/id2len.json'))
        else:
            self.id2len = {
                id_: len_
                for id_, len_ in json.load(open(f'{db_dir}/id2len.json')).items()
                if len_ <= max_txt_len
            }
        
        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)
        self.use_bert_tokenizer = use_bert_tokenizer
        
        if use_bert_tokenizer:
            self.cls_ = 101
            self.sep = 102
            self.mask = 103
        else:
            self.cls_ = 0
            self.sep = 1
            self.unknown = 2
            self.pad = 3

    def __getitem__(self, id_):
      # Fetch the raw text entry from the database
      txt_dump = self.db[id_]

      # If using BERT tokenizer, expect a pre-tokenized format
      if self.use_bert_tokenizer:
          # Make sure txt_dump is a dictionary with 'input_ids' key
          if isinstance(txt_dump, dict):
              return {"input_ids": txt_dump.get("input_ids", [])}
          else:
              raise TypeError(f"Expected dict but got {type(txt_dump)} for ID {id_}")

      # If not using BERT tokenizer, treat the data as raw text
      if isinstance(txt_dump, str):
          # Convert raw text to token IDs
          input_ids = [ord(c) for c in txt_dump[:self.fix_size - 1]]  # Simple character-based tokenizer
          input_ids.append(self.sep)  # Add sep token

          # Pad to the fixed size
          if len(input_ids) < self.fix_size:
              input_ids += [self.pad] * (self.fix_size - len(input_ids))

          return {"input_ids": input_ids}
      else:
          raise TypeError(f"Expected str but got {type(txt_dump)} for ID {id_}")


    @property
    def txt2img(self):
        if self._txt2img is None:
            txt2img_path = f'{self.db_dir}/txt2img.json'
            if not exists(txt2img_path):
                raise FileNotFoundError(f"txt2img.json not found at {txt2img_path}")
            with open(txt2img_path, 'r') as f:
                self._txt2img = json.load(f)
        return self._txt2img

    @property
    def img2txts(self):
        img2txts_path = f'{self.db_dir}/img2txts.json'
        if not exists(img2txts_path):
            raise FileNotFoundError(f"img2txts.json not found at {img2txts_path}")
        with open(img2txts_path, 'r') as f:
            return json.load(f)

class ImageLmdbGroup(object):
    def __init__(self, npy_feature=True, unconditional=False):
        self.path2imgdb = {}
        self.npy_feature = npy_feature
        self.unconditional = unconditional

    def get_dataset_name(self, path):
        # Automatically detect dataset name based on path
        if 'flickr' in path:
            return 'flickr'
        elif 'CC' in path:
            return 'cc'
        elif 'pix2code' in path:
            return 'pix2code'
        else:
            raise ValueError(f"Unknown dataset for path: {path}")

    def __getitem__(self, path):
        # Cache the database for faster lookup
        if path not in self.path2imgdb:
            dataset_name = self.get_dataset_name(path)
            self.path2imgdb[path] = DetectFeatLmdb(
                img_dir=path,
                dataset=dataset_name,
                npy_feature=self.npy_feature,
                unconditional=self.unconditional
            )
        return self.path2imgdb[path]

from torchvision.transforms import Resize, RandAugment

class DetectFeatLmdb(object):
    def __init__(self, img_dir, dataset, npy_feature=False, unconditional=False):
        self.img_dir = img_dir
        self.resize = Resize((224, 224))
        self.randaug = RandAugment(2, 10)
        self.dataset = dataset
        self.npy_feature = npy_feature
        self.unconditional = unconditional

    def __getitem__(self, file_name):
        # Ensure file name has the correct extension
        if not file_name.endswith('.png') and not file_name.endswith('.jpg') and not file_name.endswith('.npy'):
            file_name += '.png'  # Default to .png if no extension is given

        img_path = f"{self.img_dir}/{file_name}"
        
        try:
            # if self.npy_feature:
            #     img = np.load(img_path, allow_pickle=True)
            # else:
            img = Image.open(img_path).convert("RGB")
            img = self.resize(img)
            img = self.randaug(img)
            img = transforms.ToTensor()(img)
        except Exception as e:
            raise ValueError(f"Error loading image file {img_path}: {e}")

        return img



class ItmRankDataset(Dataset):
    def __init__(self, txt_db, img_db, config):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.txt2img = self.txt_db.txt2img
        self.ids = list(self.txt2img.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
      txt_id = self.ids[i]
      img_id = self.txt2img[txt_id]

      # Fetch text and image features
      inputs = self.txt_db[txt_id]
      img_feat = self.img_db[img_id]

      # Expecting 4 items here
      input_ids = inputs["input_ids"]
      loss_mask = inputs.get("loss_mask", [1] * len(input_ids))  # Default to ones if not present
      return input_ids, img_feat, loss_mask, img_id
