import os
import json

def generate_txt2img(txt_dir, output_path):
    txt2img = {}
    for filename in os.listdir(txt_dir):
        if filename.endswith(".gui"):
            # Use the full filename as the ID, including the .gui extension
            txt_id = filename
            # Assuming the image ID is the same as the file name without the extension
            img_id = txt_id.replace(".gui", "")
            txt2img[txt_id] = img_id

    # Check if the directory exists before writing the file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(txt2img, f, indent=4)
    print(f"✅ txt2img.json generated successfully at {output_path}")

if __name__ == "__main__":
    # Generate txt2img.json for train_texts
    train_texts_dir = "./data/train_texts"
    val_texts_dir = "./data/val_texts"

    # Generate for train
    output_file = os.path.join(train_texts_dir, "txt2img.json")
    generate_txt2img(train_texts_dir, output_file)

    # Generate for val
    output_file = os.path.join(val_texts_dir, "txt2img.json")
    generate_txt2img(val_texts_dir, output_file)

