# download_dataset.py
import os
import urllib.request
import tarfile

def download_dataset(url, destination):
    os.makedirs(destination, exist_ok=True)

    # Download and extract the dataset
    file_path = os.path.join(destination, "VOCdevkit.tar.gz")
    urllib.request.urlretrieve(url, file_path)
    
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(destination)

if __name__ == "__main__":
    dataset_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    download_destination = "VOCdevkit"
    
    download_dataset(dataset_url, download_destination)
