from django.contrib.auth.models import AbstractUser
from django.db import models

# Create your models here.
class User(AbstractUser):

    GENDER_MALE = "male"
    GENDER_FEMALE = "female"
    GENDER_OTHER = "other"

    GENDER_CHOICES = (
        (GENDER_MALE, "Male"),
        (GENDER_FEMALE, "Female"),
        (GENDER_OTHER, "Other"),
    )

    # 기본적으로, 미디어는 uploads 폴더에 들어간다.
    # upload_to는 그 폴더 안에서 또 어디에 위치시킬지에 관한 것이다.
    avatar = models.ImageField(upload_to="avatars", null=True, blank=True)
    gender = models.CharField(choices=GENDER_CHOICES,
                              max_length=10, null=True, blank=True)
    superhost = models.BooleanField(default=False)
    superhost = models.BooleanField(default=False)