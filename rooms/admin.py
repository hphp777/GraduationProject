from django.contrib import admin
from django.utils.html import mark_safe
from . import models

# Register your models here.


@admin.register(models.RoomType, models.Facility, models.Amenity, models.HouseRule)
class ItemAdmin(admin.ModelAdmin):

    """ Item Admin Definition """

    list_display = ("name", "used_by")

    def used_by(self, obj):
        return obj.rooms.count()

# admin 안에 admin을 추가해주는 방식


class PhotoInline(admin.TabularInline):

    model = models.Photo


@admin.register(models.Room)
class RoomAdmin(admin.ModelAdmin):

    """ Room Admin Definition """

    inlines = (PhotoInline, )

    # room을 추가할 때 화면에 표시해주는 방식
    fieldsets = (
        (
            "Basic Info",
            {"fields": ("name", "description", "country", "address", "price")}
        ),
        (
            "Times",
            {"fields": ("check_in", "check_out", "instant_book")}
        ),
        (
            "Spaces",
            {"fields": ("guests", "bed", "bedrooms", "baths")}
        ),
        (
            "More About the Spaces",
            {
                "classes": ("collapse",),  # 여기에 해당하는 옵션을 숨기거나 접을 수 있게 만들어줌
                "fields": ("Amenities", "Facilities", "house_rules")
            }
        ),
        (
            "Last Details",
            {"fields": ("host",)}
        )
    )

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
        "count_amenities",  # 함수로 만들어준 것을 디스플레이. 이거는 클릭해도 정렬이 안됨
        "count_photos",
        "total_rating",
    )

    ordering = ("name", "price", "bedrooms")

    # 객체를 거를 것들
    list_filter = (
        "instant_book",
        "host__superhost",
        "city",
        "country",
        "room_type",
        "Amenities",
        "Facilities",
        "house_rules",
    )

    # many to many에서 작동하는 필터
    filter_horizontal = ("Amenities", "Facilities", "house_rules")

    # foreign key를 더 나은 방법으로 찾을 수 있는 방법을 제공
    raw_id_fields = ("host", )

    # 이는 대소문자를 구분하지 않음. 검색되는 것은 도시.
    # foreign key에 접근하는 방법 == __
    search_fields = ['city', 'host__username']

    # many-to-many를 display해주기 위해서는 function을 만들 필요가 있다.
    # room 객체와 열을 입력 받음
    def count_amenities(self, obj):
        return obj.Amenities.count()

    def count_photos(self, obj):
        return obj.photos.count()
    count_photos.short_decription = "Photo Count"


@admin.register(models.Photo)
class PhotoAdmin(admin.ModelAdmin):

    """ Photo Admin Definition """

    list_display = ('__str__', 'get_thumbnail')

    def get_thumbnail(self, obj):
        # django에게 해당 url로 접근해서 파일을 받아도 안전하다고 알려줌
        return mark_safe(f'<img width="50px" src = "{obj.file.url}" />')
    get_thumbnail.short_description = "Thumbnail"
