from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from io import BytesIO
from common import *
import soundfile
import os
import torch
from pydub import AudioSegment
import tempfile
from FaceNet.common import *
from typing import List, Optional
from video_processor import VideoProcessor
from contextlib import asynccontextmanager
import asyncio


VideoProcessor = VideoProcessor()
# recorder = VideoRecorder()
# recorder.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("启动应用，初始化数据库...")
        await asyncio.to_thread(create_db_tables)
        print("数据库和表格已成功创建。")
    except Exception as e:
        print(f"初始化数据库时出错: {e}")
        raise e
    yield

app = FastAPI(title="人脸与语音识别 API", lifespan=lifespan)


# 加载声纹模型
device = torch.device('cpu')
audio_model_path = r'model/weights/baseline_lite_ap.model'
audio_model = load_model(audio_model_path, device)

# 加载人脸模型
# face_model_path = r'FaceNet\models\mobileface.ckpt'
# face_model = load_recognition_model(face_model_path)
# 视屏录制
#video_processor = VideoProcessor(model_path=face_model_path)

# 响应模型
# class FeatureResponse(BaseModel):
#     feature_id: int
#     feature_type: str
#     similarity: float
#
# class VideoProcessResponse(BaseModel):
#     results: List[FeatureResponse]


class SimilarityResult(BaseModel):
    id: int
    similarity: float

class FeatureResponse(BaseModel):
    feature_type: str
    top_similarities: Optional[List[SimilarityResult]] = None
    inserted_feature_id: Optional[int] = None
    inserted_similarity: Optional[float] = None
    NoPeople: Optional[str] = None

# 上传音频文件接口
@app.post("/upload_audio", response_model=FeatureResponse)
async def upload_audio(file: UploadFile = File(...)):
    # 阈值
    similarity_threshold = -0.65
    feature_type = 'audio'

    try:
        audio_bytes = await file.read()
        audio_io = BytesIO(audio_bytes)
        audio_format = file.filename.split('.')[-1].lower()
        if audio_format != 'wav':
            audio = AudioSegment.from_file(audio_io, format=audio_format)
            wav_io = BytesIO()
            audio.export(wav_io, format='wav')
            wav_io.seek(0)
            audio_bytes = wav_io.read()

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
            temp_audio_path = temp_audio_file.name
            temp_audio_file.write(audio_bytes)

        embedding = get_embedding(model=audio_model, audio_path=temp_audio_path, device=device)
        os.remove(temp_audio_path)
        top_similarities = compare_audio_features(embedding,device, top_k=2)


        response = FeatureResponse(
            feature_type = feature_type,
            top_similarities = []
        )

        if not top_similarities:
            new_id = insert_audio_feature_to_db(embedding)
            response.inserted_feature_id = new_id
            response.inserted_similarity = 1.0

        else:
            # 填充相似度列表
            response.top_similarities = [SimilarityResult(id=_id, similarity=sim) for _id, sim in top_similarities]

            top_id, top_sim = top_similarities[0]

            if top_sim > similarity_threshold:
                pass
            else:
                new_id = insert_audio_feature_to_db(embedding)
                response.inserted_feature_id = new_id
                response.inserted_similarity = 1.0
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'An error occurred: {e}')





@app.post("/upload_video", response_model=FeatureResponse)
async def upload_video(file: UploadFile = File(...)):
    similarity_threshold = 0.45
    feature_type = 'video'
    try:
        video_bytes = await file.read()

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video_file:
            temp_video_path = temp_video_file.name
            temp_video_file.write(video_bytes)

        embedding = VideoProcessor.get_most_talking_person(temp_video_path)
        os.remove(temp_video_path)
        if embedding is None:
            response = FeatureResponse(
                feature_type = feature_type,
                NoPeople="No people speaking"
            )
            return response

        else:
            top_similarities = compare_face_features(embedding, device, top_k=2)
            response = FeatureResponse(
                feature_type = feature_type,
                top_similarities=[]
            )
            if not top_similarities:
                new_id = insert_face_features_to_db(embedding)
                response.inserted_feature_id = new_id
                response.inserted_similarity = 1.0

            else:
                response.top_similarities = [SimilarityResult(id=id, similarity=sim) for id, sim in top_similarities]
                top_id, top_sim = top_similarities[0]

                if top_sim > similarity_threshold:
                    pass
                else:
                    new_id = insert_face_features_to_db(embedding)
                    response.inserted_feature_id = new_id
                    response.inserted_similarity = 1.0
            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'An error occurred: {e}')





@app.post("/process_face_video", response_model=FeatureResponse)
async def process_face_video(
        start_time: float = Form(...),
        end_time: float = Form(...),):
    similarity_threshold = 0.45
    feature_type = 'video'
    try:
        video_path = VideoProcessor.merge_video_segments(T_start=start_time, T_end=end_time)
        print(1132146)
        embedding = VideoProcessor.get_most_talking_person(video_path)

        if embedding is None:
            response = FeatureResponse(
                feature_type = feature_type,
                NoPeople="No people speaking"
            )
            return response

        else:
            top_similarities = compare_face_features(embedding, device, top_k=2)
            response = FeatureResponse(
                feature_type = feature_type,
                top_similarities=[]
            )
            if not top_similarities:
                new_id = insert_face_features_to_db(embedding)
                response.inserted_feature_id = new_id
                response.inserted_similarity = 1.0

            else:
                response.top_similarities = [SimilarityResult(id=id, similarity=sim) for id, sim in top_similarities]
                top_id, top_sim = top_similarities[0]

                if top_sim > similarity_threshold:
                    pass
                else:
                    new_id = insert_face_features_to_db(embedding)
                    response.inserted_feature_id = new_id
                    response.inserted_similarity = 1.0
            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'An error occurred: {e}')








