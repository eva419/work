"""
URL configuration for medixpert project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework.routers import DefaultRouter
from rest_framework.authtoken.views import obtain_auth_token

# Create a router for API endpoints
router = DefaultRouter()

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # API Authentication
    path('api/auth/token/', obtain_auth_token, name='api_token_auth'),
    
    # API endpoints
    path('api/', include(router.urls)),
    path('api/patients/', include('patients.urls')),
    path('api/doctors/', include('doctors.urls')),
    path('api/appointments/', include('appointments.urls')),
    path('api/ml/', include('ml_models.urls')),
    path('api/reports/', include('reports.urls')),
    path('api/chat/', include('chat.urls')),
    
    # DRF browsable API
    path('api-auth/', include('rest_framework.urls')),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# Admin site customization
admin.site.site_header = "MediXpert Administration"
admin.site.site_title = "MediXpert Admin Portal"
admin.site.index_title = "Welcome to MediXpert Administration"

