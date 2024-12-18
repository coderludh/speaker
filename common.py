import sys
import os
from audioop import reverse
import torch
from model.ResNetSE34L import MainModel
import torch.nn.functional as F
import random
import soundfile
from pydub import AudioSegment
from io import BytesIO
import sqlite3
from FaceNet.common import *


device = torch.device('cpu')

# 加载模型权重
def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    try:
        audio, sample_rate = soundfile.read(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = np.array([int(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = np.stack(feats, axis=0).astype(float)
    feat = torch.FloatTensor(feat)
    return feat

def load_model(model_path, device='cpu'):
    # 如果打包后运行，则路径可能需要调整
    if getattr(sys, 'frozen', False):
        # 运行在PyInstaller打包的环境
        base_path = sys._MEIPASS
    else:
        # 运行在开发环境
        base_path = os.path.dirname(os.path.abspath(__file__))

    full_model_path = os.path.join(base_path, model_path)

    if not os.path.isfile(full_model_path):
        print(f"模型文件不存在: {full_model_path}")
        sys.exit(1)

    model = MainModel()
    try:
        state_dict = torch.load(full_model_path, map_location=device)
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        sys.exit(1)

    new_state_dict = {}
    # 遍历原始 state_dict，将 __S__ 前缀移除
    for key, value in state_dict.items():
        # 如果键包含 __S__ 前缀，移除它
        if key.startswith("__S__."):
            new_key = key.replace("__S__.", "")
        else:
            new_key = key
        # 将修改后的键值对存储到新的 state_dict 中
        new_state_dict[new_key] = value

    try:
        # 然后将 new_state_dict 应用于模型
        model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"Error loading state_dict into model: {e}")
        sys.exit(1)

    model.eval()
    model.to(device)
    return model

def get_embedding(model, audio_path, device='cpu'):
    processed_spectrogram = loadWAV(filename=audio_path, max_frames=400)
    input_tensor = processed_spectrogram.to(device)
    # print("input_tensor", input_tensor)
    with torch.no_grad():
        # 不增加任何维度
        # input_tensor = input_tensor.unsqueeze(0)
        # print(f"Input tensor shape: {input_tensor.dim()}")  # 打印输入张量的形状
        embedding = model(input_tensor)
        embedding = F.normalize(embedding, p=2, dim=1)
        # print("embedding shape", embedding)
    return embedding

def compute_similarity(emb1, emb2):

    dist = torch.cdist(emb1.reshape(10, -1), emb2.reshape(10, -1)).detach().cpu().numpy()
    similarity = -1 * np.mean(dist)  # 取负距离作为相似度
    return similarity


def convert_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file)
    wav_io = BytesIO()
    audio.export (wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 定义 tem 文件夹路径
TEM_DIR = os.path.join(BASE_DIR, "tem")

# 如果 tem 文件夹不存在，则创建它
os.makedirs(TEM_DIR, exist_ok=True)

# 定义数据库文件路径
DATABASE_FILE = os.path.join(TEM_DIR, "feature.db")

# 获取数据库连接
def get_db_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

# 创建数据库表格
def create_db_tables():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 创建声纹特征表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audio_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature BLOB NOT NULL
        );
    """)

    # 创建人脸特征表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature BLOB NOT NULL
        );
    """)

    conn.commit()
    conn.close()

 # 插入声纹特征到数据库
def insert_audio_feature_to_db(embedding):
    conn = get_db_connection()
    cursor = conn.cursor()
    embedding_bytes = embedding.cpu().numpy().astype(np.float32).tobytes()
    cursor.execute("INSERT INTO audio_features (feature) VALUES (?)", (embedding_bytes,))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id

# 插入人脸特征
def insert_face_features_to_db(embedding):
    conn = get_db_connection()
    cursor = conn.cursor()
    embedding_bytes = embedding.cpu().numpy().astype(np.float32).tobytes()
    cursor.execute("INSERT INTO face_features (feature) VALUES(?)", (embedding_bytes,))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id

# 与数据库中的声纹特征进行比较
def compare_audio_features(embedding, device=device, top_k=2):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audio_features")
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        return None

    similarities = []

    for row in rows:
        stored_embedding = np.frombuffer(row['feature'], dtype=np.float32)
        stored_embedding = torch.from_numpy(stored_embedding).unsqueeze(0).to(device)
        similarity = compute_similarity(embedding, stored_embedding)
        similarities.append((row['id'], similarity))

    conn.close()
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar = similarities[:min(top_k, len(similarities))]

    return top_similar


def compute_face_similarity(feature1, feature2):
    feature1 -= np.mean(feature1, axis=1, keepdims=True)
    feature1 /= np.linalg.norm(feature1, axis=1, keepdims=True)
    feature2 -= np.mean(feature2, axis=1, keepdims=True)
    feature2 /= np.linalg.norm(feature2, axis=1, keepdims=True)
    # Compute similarity (cosine similarity)
    similarity = np.dot(feature1, feature2.T)
    # Since features are 1 x N, similarity is a scalar
    similarity = similarity[0][0]
    return similarity

# 比较面部特征与数据库中最相似的特征
def compare_face_features(embedding, device=device, top_k=2):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM face_features")
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        return None, None

    similarities = []

    for row in rows:
        storred_embedding = np.frombuffer(row['feature'], dtype=np.float32)
        storred_embedding = torch.from_numpy(storred_embedding).unsqueeze(0).to(device)
        similarity = compute_face_similarity(embedding, storred_embedding)
        similarities.append((row['id'], similarity))

    conn.close()
    # 降序排列
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar = similarities[:min(top_k, len(similarities))]

    return top_similar








