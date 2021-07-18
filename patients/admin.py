from django.contrib import admin
from django.utils.html import mark_safe
from . import models

# Register your models here.


class ImageInline(admin.TabularInline):
    model = models.Image


@admin.register(models.Patient)
class PatientAdmin(admin.ModelAdmin):

    """Patient Admin Definition"""

    inlines = (ImageInline,)

    list_display = (
        "name",
        "id",
        "age",
        "gender",
        "doctor",
        "count_images",
    )

    ordering = ("id",)

    list_filter = ("doctor",)

    search_fields = ("=id", "^name")

    def count_images(self, obj):
        return obj.images.count()


@admin.register(models.Image)
class ImageAdmin(admin.ModelAdmin):

    """Image Admin Definition"""

    list_display = ("__str__", "get_thumbnail")

    def get_thumbnail(self, obj):
        return mark_safe(f'<img width="50px" src="{obj.file.url}" />')

    get_thumbnail.short_description = "Thumbnail"