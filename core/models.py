from django.db import models

# Create your models here.
# 이 모델은 다른 모델들의 common field를 가지는 모델.
# 하지만 이 모델이 데이터베이스에 등록되기는 원하지 않음.


class TimeStampedModel(models.Model):

    """Time Stamped Model"""

    # auto_now_add는 모델을 만들 때마다 현재의 시각을 기록해주는 옵션
    # auto_now는 모델을 저장 할 때마다 현재의 시각을 기록해주는 옵션
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    # 클래스에 관한 기타 사항을 적을 수 있는 클래스
    class Meta:
        # abstract란? 데이터베이스에 나타나지 않는 모델
        # 이 모델을 다른 모델에서 확장해서 사용한다는 뜻이다.
        abstract = True
