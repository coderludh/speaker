import torch
from PIL import Image
import numpy as np
from rknnlite.api import RKNNLite


rknn = RKNNLite(verbose=True)

ret = rknn.load_rknn(path='./FaceNet/models/mobilefacenet.rknn')
ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
def preprocess_image(image_path):
    """
    Preprocesses the image by loading it, converting to RGB if necessary,
    normalizing, and preparing it for input into the model.
    """
    img = Image.open(image_path)
    img = np.array(img)
    img = img.astype(np.float32)

    if len(img.shape) == 2:  # If grayscale, convert to RGB
        img = np.stack([img]*3, axis=2)
    # Create a list with original and horizontally flipped images
    img_list = [img, img[:, ::-1, :]]
    for i in range(len(img_list)):
        # Normalize the image
        img_list[i] = (img_list[i] - 127.5) / 128.0
        # Transpose to (channel, height, width)
        img_list[i] = img_list[i].transpose(2, 0, 1)
        img_list[i] = np.expand_dims(img_list[i], axis=0)
    return img_list

def extract_feature(imgs):
    """
    Passes the preprocessed images through the network to extract features.
    """
    features = []
    for i in range(len(imgs)):
        feature = rknn.inference(inputs=[imgs[i]], data_format='nchw')
        features.append(feature[0])
    feature = np.concatenate((features[0], features[1]), axis=1)
    return feature

def compare_images(image_path1, image_path2, threshold=0.5):
    """
    Compares two images to determine if they are of the same person.
    """
    imgs1 = preprocess_image(image_path1)
    imgs2 = preprocess_image(image_path2)
    # Extract features
    feature1 = extract_feature(imgs1)
    feature2 = extract_feature(imgs2)
    print(f"Feature1 shape: {feature1.shape}")  # 例如 (1, 512)
    print(f"Feature1: {feature1}")
    print(f"Feature2 shape: {feature2.shape}")  # 例如 (1, 512)
    print(f"Feature2: {feature2}")
    # Normalize features
    feature1 -= np.mean(feature1, axis=1, keepdims=True)
    feature1 /= np.linalg.norm(feature1, axis=1, keepdims=True)
    feature2 -= np.mean(feature2, axis=1, keepdims=True)
    feature2 /= np.linalg.norm(feature2, axis=1, keepdims=True)
    # Compute similarity (cosine similarity)
    similarity = np.dot(feature1, feature2.T)
    # Since features are 1 x N, similarity is a scalar
    similarity = similarity[0][0]
    # Decide based on threshold
    if similarity > threshold:
        result = "Same person"
    else:
        result = "Different persons"
    print(f"Result: {result}")
    print(f"Similarity score: {similarity}")
    return result, similarity

if __name__ == '__main__':
    # Paths to your images
    image_path1 = r'./data/pic/Alejandro_Atchugarry_0002.jpg'
    image_path2 = r'./data/pic/Alejandro_Atchugarry_0001.jpg'
    compare_images(image_path1, image_path2)
    # image_path1 = r'./data/pic/xu.jpg'
    # image_path2 = r'./data/pic/xu2.jpg'
    # compare_images(image_path1, image_path2)
    # image_path1 = r'./data/pic/xu3.jpg'
    # image_path2 = r'./data/pic/xu2.jpg'
    # compare_images(image_path1, image_path2)
    #
    # image_path1 = r'./data/pic/Alejandro_Atchugarry_0001.jpg'
    # image_path2 = r'./data/pic/zhou2.jpg'
    # compare_images(image_path1, image_path2)
