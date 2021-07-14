from django.urls import path
from . import views

app_name = "patients"


urlpatterns = [
    path("list/", views.PatientView.as_view(), name="list"),
    path("<int:pk>", views.PatientDetail.as_view(), name="detail"),
]
