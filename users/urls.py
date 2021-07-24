from . import views
from django.urls import path

app_name = "users"

urlpatterns = [
    path("login/", views.login, name="login"),
    path("logout", views.log_out, name="logout"),
    path("sigup", views.signup, name="signup"),
]
