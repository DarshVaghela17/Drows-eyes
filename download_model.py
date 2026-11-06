import urllib.request
import os

url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
dest = 'face_landmarker.task'

if not os.path.exists(dest):
    print('Downloading face_landmarker.task...')
    urllib.request.urlretrieve(url, dest)
    print(f'Downloaded: {os.path.getsize(dest)} bytes')
else:
    print(f'Model already exists: {os.path.getsize(dest)} bytes')
