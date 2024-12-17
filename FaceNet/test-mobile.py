import torch
from PIL import Image
import numpy as np
from models.mobileface import MobileFacenet

def preprocess_image(image_path):
    """
    Preprocesses the image by loading it, converting to RGB if necessary,
    normalizing, and preparing it for input into the model.
    """
    img = Image.open(image_path)
    print(img.size)
    img = np.array(img)
    print(img.shape)

    if len(img.shape) == 2:  # If grayscale, convert to RGB
        img = np.stack([img]*3, axis=2)
    # Create a list with original and horizontally flipped images
    img_list = [img, img[:, ::-1, :]]
    for i in range(len(img_list)):
        # Normalize the image
        img_list[i] = (img_list[i] - 127.5) / 128.0
        # Transpose to (channel, height, width)
        img_list[i] = img_list[i].transpose(2, 0, 1)
    # Convert to PyTorch tensors and add batch dimension
    imgs = [torch.from_numpy(i).float().unsqueeze(0) for i in img_list]
    return imgs

def extract_feature(imgs, net):
    """
    Passes the preprocessed images through the network to extract features.
    """
    net.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            imgs = [img.cuda() for img in imgs]
        # Get features from the model
        features = [net(img).cpu().numpy() for img in imgs]
        # Concatenate features from original and flipped images
        feature = np.concatenate((features[0], features[1]), axis=1)
    return feature

def compare_images(image_path1, image_path2, model_path, threshold=0.5):
    """
    Compares two images to determine if they are of the same person.
    """
    # Load the model
    net = MobileFacenet()
    if torch.cuda.is_available():
        net = net.cuda()
    # Load model weights
    ckpt = torch.load(model_path)
    net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    # Preprocess images
    imgs1 = preprocess_image(image_path1)
    imgs2 = preprocess_image(image_path2)
    # Extract features
    feature1 = extract_feature(imgs1, net)
    feature2 = extract_feature(imgs2, net)
    print(f"Feature1 shape: {feature1.shape}")  # 例如 (1, 512)
    print(f"Feature2 shape: {feature2.shape}")  # 例如 (1, 512)
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
    image_path1 = r'.\data\detected_faces\lu2.jpg'
    image_path2 = r'.\data\detected_faces\face_1.jpg'
    # Path to your trained model
    model_path = r'.\models\mobileface.ckpt'
    # Compare the two images
    compare_images(image_path1, image_path2, model_path)
