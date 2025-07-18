from django.urls import path, include
from rest_framework.routers import DefaultRouter

app_name = 'ml_models'

# Create a router for ViewSets
router = DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
]

