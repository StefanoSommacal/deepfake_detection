from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='deepfake_detection'),
    path('predict_video/', views.predict_video, name='deepfake_detection_prediction'),
]
