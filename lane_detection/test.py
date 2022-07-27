import zipfile
import os
import shutil
# shutil.rmtree('archive')
path = 'data'
unzip_path = 'unzipped_data'
with zipfile.ZipFile('data/data.zip', 'r') as zip_ref:
    print('Unzipping file')
    zip_ref.extractall(unzip_path)

for file in os.listdir(unzip_path):
    shutil.move(os.path.join(unzip_path, file), os.path.join(path, file))
shutil.rmtree(unzip_path)
