from django.urls import path
from .views import UploadPdfView, AskPdfView, HealthView

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("pdf/upload/", UploadPdfView.as_view(), name="pdf-upload"),
    path("pdf/ask/", AskPdfView.as_view(), name="pdf-ask"),
]
