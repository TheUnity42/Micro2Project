from SQLTools import Enable_SQLite_Image_Compressor
import numpy as np
import pandas as pd
import imageio
import sqlite3
import re
import glob
import tqdm
import kaggle
import sys
import PIL


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

def ProcessJoints(dir):
    print("Processing Joint data...")
    # create a sqlite database
    db_path = './Database/joint_data.db'
    db_path_lite = './Database/joint_data_lite.db'
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn_lite = sqlite3.connect(db_path_lite, detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c_lite = conn_lite.cursor()
    # create a table for the images and labels, if it does not exist
    c.execute("DROP TABLE IF EXISTS data;")
    c.execute("CREATE TABLE data (id integer primary key, img array, label array, bbox array);")
    c_lite.execute("DROP TABLE IF EXISTS data;")
    c_lite.execute("CREATE TABLE data (id integer primary key, img array, label array, bbox array);")
    for path in tqdm.tqdm(glob.glob(f"{dir}\\annotated_frames\\data_1\\*.jpg")):
        # find the labels, in dir \\projections_2d\\data_1\\
        # 0_webcam_1.jpg -> 0_jointsCam_1.txt
        # n_webcam_k.jpg -> n_jointsCam_k.txt

        # find the bounding box, in dir \\bounding_boxes\\data_1\\
        # 0_webcam_1.jpg -> 0_bbox_1.txt
        # n_webcam_k.jpg -> n_bbox_k.txt

        # get the file name
        file_name = path.split("\\")[-1]
        # get the label name
        label_name = file_name.split("_")[0] + "_jointsCam_" + file_name.split("_")[-1].split(".")[0] + ".txt"
        # get the bounding box name
        bbox_name = file_name.split("_")[0] + "_bbox_" + file_name.split("_")[-1].split(".")[0] + ".txt"

        # read the label into a pandas dataframe
        df = pd.read_csv(f"{dir}\\projections_2d\\data_1\\{label_name}", sep=" ", header=None)
        df.columns = ['Joint', 'X', 'Y']

        # read the bounding box into a pandas dataframe
        bbox = pd.read_csv(f"{dir}\\bounding_boxes\\data_1\\{bbox_name}", sep=" ", header=None)
        bbox.columns = ['Loc', 'Val']

        # get the image
        data = np.asarray(imageio.imread(path), dtype=np.uint8)

        # convert the dataframe to a numpy array
        data_array = df.drop(columns=['Joint']).to_numpy()
        # convert the dataframe to a numpy array
        box_array = bbox.drop(columns=['Loc']).to_numpy()
        # cast box to float
        box_array = np.cast[np.float32](box_array)

        # map data to image size
        data_array[:, 0] = data_array[:, 0] / data.shape[1]
        data_array[:, 1] = data_array[:, 1] / data.shape[0]

        # map box from [Top, Left, Bottom, Right] to [Bottom, Left, Width, Height]
        bbox = np.zeros(4, dtype=np.float32)
        bbox[0] = box_array[0, 0]
        bbox[1] = box_array[1, 0]
        bbox[2] = box_array[2, 0]
        bbox[3] = box_array[3, 0]
        
        # insert the data into the database
        c.execute("INSERT INTO data VALUES (NULL, ?, ?, ?)",
                  (data, data_array, bbox))

        # reduce the images to a smaller size (224x224)
        data_lite = PIL.Image.fromarray(data)
        data_lite = data_lite.resize((224, 224))
        data_lite = np.asarray(data_lite, dtype=np.uint8)

        # adjust bbox to new image size
        bbox[0] = bbox[0] * 224 / data.shape[0]
        bbox[1] = bbox[1] * 224 / data.shape[1]
        bbox[2] = bbox[2] * 224 / data.shape[0]
        bbox[3] = bbox[3] * 224 / data.shape[1]

        # insert the data into the database
        c_lite.execute("INSERT INTO data VALUES (NULL, ?, ?, ?)", (data_lite, data_array, bbox))


    conn.commit()
    conn.close()
    conn_lite.commit()
    conn_lite.close()
    print("Wrote entries.")
    

def main(cmd):
    # activate the SQLite image compressor
    Enable_SQLite_Image_Compressor()
    
    if cmd == "install":
        ProcessKaggle()
    if cmd == "joints":
        ProcessJoints(sys.argv[2])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 setup.py <install|joints>")
        sys.exit(1)
    main(sys.argv[1])
    sys.exit(0)
