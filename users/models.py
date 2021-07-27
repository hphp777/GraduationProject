from django.contrib.auth.models import AbstractUser
from django.shortcuts import reverse
from django.db import models

# Create your models here.
# AbstractUser는 기존의 User모델을 사용하면서 추가적인 정보를 넣기 위해 사용한다.
# config settings.py에서 유저모델을 어떤걸 사용했는지 지정해주어야 한다.

class User(AbstractUser):
    """Custom user model"""

    # 데이터베이스에 들어가는 값
    GENDER_MALE = "male"
    GENDER_FEMALE = "female"
    GENDER_OTHER = "other"

    # 폼에 나타나는 값
    # 이렇게 초이스가 가능하게 해 주는것은 데이터베이스에 영향을 주지 않으므로
    # makemigrations을 해 줄 필요가 없다.
    GENDER_CHOICES = (
        (GENDER_MALE, "Male"),
        (GENDER_FEMALE, "Female"),
        (GENDER_OTHER, "Other"),
    )

    LANGUAGE_ENGLISH = "en"
    LANGUAGE_KOREAN = "kr"

    LANGUAGE_CHOICES = (
        (LANGUAGE_ENGLISH, "English"),
        (LANGUAGE_KOREAN, "Korean")
    )

    CURRENCY_USD = "usd"
    CURRENCY_KRW = "krw"

    CURRENCY_CHOICES = (
        (CURRENCY_USD, "USD"),
        (CURRENCY_KRW, "KRW")
    )

    # 기본적으로, 미디어는 uploads 폴더에 들어간다.
    # upload_to는 그 폴더 안에서 또 어디에 위치시킬지에 관한 것이다.
    avatar = models.ImageField(upload_to="avatars", null=True, blank=True)
    gender = models.CharField(choices=GENDER_CHOICES,
                              max_length=10, null=True, blank=True)
    bio = models.TextField(default="", blank=True)
    birthdate = models.DateField(null=True)
    language = models.CharField(
        choices=LANGUAGE_CHOICES, max_length=2, blank=True, default=LANGUAGE_KOREAN
    )
    currency = models.CharField(
        choices=CURRENCY_CHOICES, max_length=3, blank=True, default=CURRENCY_KRW
    )
    superhost = models.BooleanField(default=False)
    superhost = models.BooleanField(default=False)

    def get_absolute_url(self):
        return reverse('users:profile', kwargs={'pk': self.pk})
