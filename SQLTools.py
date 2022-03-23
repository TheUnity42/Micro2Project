import sqlite3
import numpy as np
import zlib
import io
import tensorflow as tf

def Enable_SQLite_Image_Compressor():    
    def adapt_array(arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)  
        return sqlite3.Binary(zlib.compress(out.read()))

    def convert_array(text):
        out = io.BytesIO(zlib.decompress(text, zlib.MAX_WBITS | 32))
        out.seek(0)
        return np.load(out)

    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)


def Extract_TF_Dataset(db, map_func=None):
    Enable_SQLite_Image_Compressor()
    conn = sqlite3.connect(db, detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute("SELECT img, label FROM data;")
    data = c.fetchall()
    conn.close()

    imgs, labels = map(list, zip(*data))
    imgs = np.asarray(imgs, dtype=np.uint8)

    if map_func is not None:
        labels = list(map(map_func, labels))

    # Create a dataset of images and labels
    dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))

    return dataset

# class SQLiteImageLoader:
#     def __init__(self, db_path, zip_mode=True):
#         self.db_path = db_path
#         self.zip_mode = zip_mode
#         self.__setup_sqlite()

#     def __adapt_array(self, arr):
#         out = io.BytesIO()
#         np.save(out, arr)
#         out.seek(0)        
#         if not self.zip_mode:
#             return sqlite3.Binary(out.read())
#         else:
#             return sqlite3.Binary(zlib.compress(out.read()))

#     def __convert_array(self, text):
#         if not self.zip_mode:
#             out = io.BytesIO(text)
#         else:
#             out = io.BytesIO(zlib.decompress(text, zlib.MAX_WBITS | 32))
#         out.seek(0)
#         return np.load(out)

#     def save_to_sqlite(self, data, table_name):
#         conn = sqlite3.connect(
#             self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
#         c = conn.cursor()
#         c.execute("DROP TABLE IF EXISTS {};".format(table_name))
#         c.execute("CREATE TABLE {} (data array);".format(table_name))
#         for i in data:
#             c.execute("INSERT INTO {} VALUES (?)".format(table_name), (i,))
#         conn.commit()
#         conn.close()

#     def read_from_sqlite(self, table_name):
#         conn = sqlite3.connect(
#             self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
#         c = conn.cursor()
#         c.execute("SELECT * FROM {}".format(table_name))
#         data = np.array(c.fetchall())[:, 0, :, :]
#         conn.close()
#         return data

#     def save_data_to_sqlite(self, data, labels, table_name, data_dtype='array', label_dtype='array'):
#         conn = sqlite3.connect(
#             self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
#         c = conn.cursor()
#         c.execute("DROP TABLE IF EXISTS {};".format(table_name))
#         c.execute("CREATE TABLE {} (id integer primary key, data {}, labels {});".format(table_name, data_dtype, label_dtype))
#         for i in range(len(data)):
#             c.execute("INSERT INTO {} VALUES (NULL, ?, ?)".format(table_name), (data[i], labels[i]))
#         conn.commit()
#         conn.close()

#     def get_length(self, table_name):
#         conn = sqlite3.connect(
#             self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
#         c = conn.cursor()
#         c.execute("SELECT COUNT(*) FROM {}".format(table_name))
#         l = c.fetchall()[0][0]
#         conn.close()
#         return l


#     def make_table(self, table_name, data_dtype='array', label_dtype='array'):
#         conn = sqlite3.connect(
#             self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
#         c = conn.cursor()
#         c.execute("DROP TABLE IF EXISTS {};".format(table_name))
#         c.execute("CREATE TABLE {} (id integer primary key, data {}, labels {});".format(table_name, data_dtype, label_dtype))
#         conn.commit()
#         conn.close()

#     def add_entry(self, data, label, table_name):
#         conn = sqlite3.connect(
#             self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
#         c = conn.cursor()
#         c.execute("INSERT INTO {} VALUES (NULL, ?, ?)".format(table_name), (data, label))
#         conn.commit()
#         conn.close()

#     def add_entry_array(self, data, label, table_name):
#         conn = sqlite3.connect(
#             self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
#         c = conn.cursor()
#         assert len(data) == len(label)
#         for i in range(len(data)):
#             c.execute("INSERT INTO {} VALUES (NULL, ?, ?)".format(table_name), (data[i], label[i]))        
#         conn.commit()
#         conn.close()

#     def read_array_data(self, table_name):
#         conn = sqlite3.connect(
#             self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
#         c = conn.cursor()
#         c.execute("SELECT * FROM {}".format(table_name))
#         data = []
#         labels = []

#         for row in c.fetchall():
#             data.append(np.array(row[1]))
#             labels.append(np.array(row[2]))

#         conn.close()
#         return np.array(data), np.array(labels)
    
#     def read_array_data_range(self, table_name, rng):
#         conn = sqlite3.connect(
#             self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
#         c = conn.cursor()
#         c.execute("SELECT * FROM {} WHERE id BETWEEN {} AND {}".format(table_name, rng[0], rng[1]))
#         data = []
#         labels = []

#         for row in c.fetchall():
#             data.append(np.array(row[1]))
#             labels.append(np.array(row[2]))

#         conn.close()
#         return np.array(data), np.array(labels)

#     def read_array_data_at(self, table_name, loc):
#         conn = sqlite3.connect(
#             self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
#         c = conn.cursor()
#         c.execute("SELECT * FROM {} WHERE id IN {}".format(table_name, tuple(loc)))
#         data = []
#         labels = []

#         for row in c.fetchall():
#             data.append(np.array(row[1]))
#             labels.append(np.array(row[2]))

#         conn.close()
#         return np.array(data), np.array(labels)


#     def __setup_sqlite(self):
#         sqlite3.register_adapter(np.ndarray, self.__adapt_array)
#         sqlite3.register_converter("array", self.__convert_array)