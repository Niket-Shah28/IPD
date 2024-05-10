from django.urls import path
from . import views

urlpatterns = [
  path('get_data/', views.get_lyrics, name='get_lyrics'),
  path('get_video_feed/', views.get_video_feed, name='get_video_feed'),
]
