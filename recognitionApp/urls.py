from django.contrib import admin
from django.urls import path, include
from . import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('deepfake_detection/admin/', admin.site.urls),
    path('deepfake_detection/', views.home, name='Home'),
    path('deepfake_detection/about/', views.about, name='About'),
    path('deepfake_detection/scanner/', include('deepfake_detection.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)   

