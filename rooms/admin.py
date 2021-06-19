from django.contrib import admin
from . import models

# Register your models here.


@admin.register(models.RoomType, models.Facility, models.Amenity, models.HouseRule)
class ItemAdmin(admin.ModelAdmin):

    """ Item Admin Definition """

    pass


@admin.register(models.Room)
class RoomAdmin(admin.ModelAdmin):

    """ Room Admin Definition """
    # room page에 들어가면 생성된 객체의 정보중 화면에 띄워줄 것들
    list_display = (
        "name",
        "price",
        "guests",
        "baths",
        "check_in",
        "check_out",
        "instant_book",
        "bed",
        "bedrooms",
        "city",
        "country",
    )

    # 객체를 거를 것들
    list_filter = ("instant_book", "city", "country", "room_type",
                   "Amenities", "Facilities", "house_rules",)

    # 이는 대소문자를 구분하지 않음. 검색되는 것은 도시.
    # foreign key에 접근하는 방법 == __
    search_fields = ['city', 'host__username']


@admin.register(models.Photo)
class PhotoAdmin(admin.ModelAdmin):

    """ Photo Admin Definition """

    pass
