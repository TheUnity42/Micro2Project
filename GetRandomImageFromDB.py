from SQLTools import Enable_SQLite_Image_Compressor, Extract_TF_Dataset
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import sys

def main(db):
    print("Loading database...")

    # Enable the SQLite image compressor
    Enable_SQLite_Image_Compressor()

    # Open the database
    conn = sqlite3.connect(db, detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()

    # Get the number of images in the database
    c.execute("SELECT COUNT(*) FROM data;")
    num_images = c.fetchone()[0]

    # choose randome number from 0 to num_images
    rand_num = np.random.randint(0, num_images)

    # fetch one random image
    c.execute("SELECT img, label FROM data WHERE id = ?;", (rand_num,))
    img, label = c.fetchone()
    data = np.asarray(img)
    assert type(data) == np.ndarray

    # display the image
    plt.imshow(data)
    plt.title(label)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 GetRandomImageFromDB.py <db_path>")
        exit(1)
    # main(sys.argv[1])
    Extract_TF_Dataset(sys.argv[1])
    exit(0)