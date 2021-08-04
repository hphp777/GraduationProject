from django.conf.urls.static import static
from django.conf import settings
from django.urls import path
from . import views

app_name = "rooms"

urlpatterns = [
    # url에 첫번째 인자를 입력하면 두번째 인자를 호출한다.
    # path에 변수를 입력받을 수도 있는것. 변수타입은 int, 이름은 pk
    path("<int:pk>/", views.RoomDetail.as_view(), name="detail"),
    path("<int:pk>/photos/", views.RoomPhotosView.as_view(), name="photos"),
    path("<int:pk>/edit/", views.EditRoomView.as_view(), name="edit"),
    path("search/", views.SearchView.as_view(), name="search"),
    path(
        "<int:room_pk>/photos/<int:photo_pk>/delete/",
        views.delete_photo,
        name="delete-photo",
    ),
     path(
        "<int:room_pk>/photos/<int:photo_pk>/edit/",
        views.EditPhotoView.as_view(),
        name="edit-photo",
    ),
] 
