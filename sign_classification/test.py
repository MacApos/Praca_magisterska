import zipfile
import os
import shutil
# shutil.rmtree('archive')

path = 'data'
if not os.path.exists(path):
    with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
        print('Rozpakowywanie pliku')
        zip_ref.extractall('data')
