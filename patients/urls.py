from . import views
from django.urls import path

app_name = "patiensts"

urlpatterns = [
    path("list", views.PatientView.as_view(), name="list"),
]
