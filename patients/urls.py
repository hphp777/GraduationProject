from django.urls import path
from . import views

app_name = "patients"


urlpatterns = [
    path("list/", views.all_patient, name="list"),
    path("<int:pk>", views.PatientDetail.as_view(), name="detail"),
    path("registration/", views.registrate, name="registration"),
    # path("diagnosis/", views.DiagnosisView.as_view(), name="diagnosis"),
]
