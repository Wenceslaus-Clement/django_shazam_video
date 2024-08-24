# api/urls.py

from django.urls import path
from .views import VideoSimilarityView

urlpatterns = [
    path('video-similarity/', VideoSimilarityView.as_view(), name='video-similarity'),
]
