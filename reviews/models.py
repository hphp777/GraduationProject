from django.db import models
from core import models as core_models

# Create your models here.


class Review(core_models.TimeStampedModel):

    """Review Model Definition"""

    review = models.TextField()
    accuracy = models.IntegerField()
    communication = models.IntegerField()
    cleanliness = models.IntegerField()
    location = models.IntegerField()
    check_in = models.IntegerField()
    value = models.IntegerField()
    user = models.ForeignKey("users.USer", on_delete=models.CASCADE)
    room = models.ForeignKey(
        "rooms.Room", related_name="reviews", on_delete=models.CASCADE)

    def __str__(self):
        # 사용한 foreign key 안으로 계속 들어갈 수 있음.
        # return self.room.room.host.username
        # 또는 아래와 같이 formatting 가능
        return f"{self.review} - {self.room}"

    # model 안에서 함수 생성하기 (like admin)
    # review의 평균을 알고 싶은 경우. 이는 특정 패널에서만 사용되는 것이 아니라 광범위하게 사용 될 것임
    def rating_average(self):
        avg = (
            self.accuracy +
            self.communication +
            self.cleanliness +
            self.location +
            self.check_in +
            self.value
        ) / 6
        return round(avg, 2)
