from django.urls import path
from . import views

app_name = "patients"


urlpatterns = [
    path("list/", views.all_patient, name="list"),
    path("list/edit/", views.all_patient_edit, name="edit"),
    path("<int:pk>/", views.detail, name="detail"),
    path("list/edit/<int:pk>/delete/", views.delete_patient, name="delete-patient"),
    path("registration/", views.registrate, name="registration"),
    # path("diagnosis/", views.DiagnosisView.as_view(), name="diagnosis"),
]
