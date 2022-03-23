from SQLTools import Enable_SQLite_Image_Compressor
import numpy as np
import imageio
import sqlite3
import re
import glob
import tqdm
import kaggle
import sys


def ProcessKaggle():

    print("Downloading Kaggle data...")
    # fetch the dataset
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "drgfreeman/rockpaperscissors", "./Data/Kaggle/", unzip=True)

    print("Processing Kaggle data...")
    # create a sqlite database
    db_path = './Database/kaggle_data.db'
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    # create a table for the images and labels, if it does not exist
    c.execute("DROP TABLE IF EXISTS data;")
    c.execute("CREATE TABLE data (id integer primary key, img array, label text);")

    counts = [0, 0, 0]
    for path in tqdm.tqdm(glob.glob('.\\Data\\Kaggle\\*\\*.png')):
        if re.search("rock", path) is not None:
            t = "rock"
        elif re.search("paper", path) is not None:
            t = "paper"
        elif re.search("scissors", path) is not None:
            t = "scissors"
        else:
            continue
        data = np.asarray(imageio.imread(path), dtype=np.uint8)
        assert type(data) == np.ndarray
        c.execute("INSERT INTO data VALUES (NULL, ?, ?)", (data, t))
        if t == "rock":
            counts[0] += 1
        elif t == "paper":
            counts[1] += 1
        elif t == "scissors":
            counts[2] += 1

    conn.commit()
    conn.close()
    print(
        f"Wrote {counts[0] + counts[1] + counts[2]} entries. ({counts[0]} rock, {counts[1]} paper, {counts[2]} scissors)")



def main(cmd):
    # activate the SQLite image compressor
    Enable_SQLite_Image_Compressor()
    
    if cmd == "install":
        ProcessKaggle()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 setup.py <install>")
        sys.exit(1)
    main(sys.argv[1])
    sys.exit(0)