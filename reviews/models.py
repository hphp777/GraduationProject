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
    room = models.ForeignKey("rooms.Room", on_delete=models.CASCADE)

    def __str__(self):
        # 사용한 foreign key 안으로 계속 들어갈 수 있음.
        # return self.room.room.host.username
        # 또는 아래와 같이 formatting 가능
        return f"{self.review} - {self.room}"
