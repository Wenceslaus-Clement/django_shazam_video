from django.shortcuts import render

# Create your views here.

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16 # type: ignore
from tensorflow.keras.models import Model # type: ignore

class VideoSimilarityView(APIView):
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

    def extract_frames(self, video_path, num_frames=5):
        cap = cv2.VideoCapture(video_path)
        frames = []
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, length // num_frames)
        
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.resize(frame, (224, 224)))
        cap.release()
        return frames

    def extract_features(self, frames):
        features = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.expand_dims(frame, axis=0)
            feature = self.model.predict(frame)
            features.append(feature.flatten())
        return np.mean(features, axis=0)

    def get_video_features(self, video_path):
        frames = self.extract_frames(video_path)
        features = self.extract_features(frames)
        return features

    def get_similarity(self, features1, features2):
        return cosine_similarity([features1], [features2])[0][0]

    def post(self, request, format=None):
        video_path = request.data.get('video_path')
        video_dir = request.data.get('video_dir')
        
        input_video_features = self.get_video_features(video_path)
        similarities = []

        for video_file in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_file)
            video_features = self.get_video_features(video_path)
            similarity = self.get_similarity(input_video_features, video_features)
            similarities.append((video_file, similarity * 100))

        # Sort by similarity
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        return Response({'similarities': similarities}, status=status.HTTP_200_OK)
