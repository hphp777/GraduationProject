from django.urls import path
from django.urls.resolvers import URLPattern
from users import views as user_views

app_name = "core"

# 첫번째 인자로 url을 받았을 때 홈뷰를 띄운다.
# 띄우는 함수가 as_view


urlpatterns = [path("", user_views.login, name="home")]
