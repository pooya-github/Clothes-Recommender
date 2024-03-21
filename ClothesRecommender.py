import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.neighbors import NearestNeighbors
from google.colab.patches import cv2_imshow

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')


!cp -r /content/drive/My\ Drive/Upwork/ClothesRecommender/zalando.zip /content/
!cp -r /content/drive/My\ Drive/Upwork/ClothesRecommender/filenames.pkl /content/
!cp -r /content/drive/My\ Drive/Upwork/ClothesRecommender/embeddings.pkl /content/

filenames = pickle.load(open('/content/filenames.pkl','rb'))
feature_list= pickle.load(open('/content/embeddings.pkl','rb'))

!unzip zalando.zip -d zalando

# Function to read and display the image
def read_and_display_uploaded_image(uploaded_image):
    # Convert the uploaded image bytes to a numpy array
    nparr = np.frombuffer(uploaded_image, np.uint8)

    # Decode the numpy array into an OpenCV image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display using matplotlib
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# Upload the image file(s)
uploaded = files.upload()

# Assuming there's only one uploaded image, read and display it
for filename, content in uploaded.items():
    print('Uploaded image is: ', filename)
    read_and_display_uploaded_image(content)
    print('Clothes similar to above are: ')
    img = image.load_img(f'/content/{filename}',target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list)

    distances,indices = neighbors.kneighbors([normalized_result])

    for file in indices[0][1:6]:
        temp_img = cv2.imread(filenames[file])
        cv2_imshow(cv2.resize(temp_img,(512,512)))