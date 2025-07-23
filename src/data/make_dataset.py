import os
import urllib.request
import zipfile
import pandas as pd

def fetch_data(path, url):

    os.makedirs(path, exist_ok=True)
    archive_zip = os.path.join(path, "archive.zip")
    urllib.request.urlretrieve(url, archive_zip)
    try :
        with zipfile.ZipFile(archive_zip, "r") as zip_ref:
            zip_ref.extractall(path)
            print("Files extracted successfully.")
            print(f"path : {path}")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")

def load_data(path, nameOFfile = "train.csv"):
    return (pd.read_csv(os.path.join(path, nameOFfile)))