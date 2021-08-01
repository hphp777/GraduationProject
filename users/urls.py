from . import views
from django.urls import path

app_name = "users"

urlpatterns = [
    path("login/", views.login, name="login"),
    path("logout", views.log_out, name="logout"),
    path("sigup", views.signup, name="signup"),
    path("<int:pk>/", views.UserProfileView.as_view(), name="profile"),
    path("update-profile/", views.UpdateProfileView.as_view(), name="update"),
    path("update-password/", views.UpdatePasswordView.as_view(), name="password")
]
