from django.urls import path
from . import views

app_name = "patients"


urlpatterns = [
    path("list/", views.all_patient, name="list"),
    path("list/edit/", views.all_patient_edit, name="edit"),
    path("list/edit/<int:pk>/delete/", views.delete_patient, name="delete-patient"),
    path("<int:pk>/", views.detail, name="detail"),
    path(
        "<int:patient_pk>/<int:image_pk>/delete/",
        views.delete_image,
        name="delete-image",
    ),
    path("registration/", views.registrate, name="registration"),
]
