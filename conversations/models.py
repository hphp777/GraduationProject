from django.core.checks import messages
from django.db import models
from core import models as core_models

# Create your models here.


class Conversation(core_models.TimeStampedModel):

    """ Conversation Model Definition """

    participants = models.ManyToManyField(
        "users.USer", related_name="conversation", blank=True)

    def __str__(self):
        usernames = []
        for user in self.participants.all():
            usernames.append(user.username)  # 객체를 추가하는게 아니라 이름을 추가해야 함.
        title = ", ".join(usernames)
        print(title)
        return title

    def count_messages(self):
        return self.messages.count()
    count_messages.short_description = "Number of Messages"

    def count_participants(self):
        return self.participants.count()
    count_participants.short_description = "Number of participants"


class Message(core_models.TimeStampedModel):

    message = models.TextField()
    user = models.ForeignKey(
        "users.User", related_name="messages", on_delete=models.CASCADE)
    conversation = models.ForeignKey(
        "Conversation", related_name="messages", on_delete=True)

    # message 객체가 표현되는 방식
    # 딱히 display를 해주지 않음
    def __str__(self):
        return f"{self.user} says: {self.message}"
