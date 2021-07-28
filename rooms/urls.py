from django.conf.urls.static import static
from django.conf import settings
from django.urls import path
from . import views

app_name = "rooms"

urlpatterns = [
    # url에 첫번째 인자를 입력하면 두번째 인자를 호출한다.
    # path에 변수를 입력받을 수도 있는것. 변수타입은 int, 이름은 pk
    path("<int:pk>/", views.RoomDetail.as_view(), name="detail"),
    path("search/", views.SearchView.as_view(), name="search"),
] 
