from django.core.management.base import BaseCommand
from rooms.models import Facility

# 모델이 아니라 인스턴스, 즉 객체를 생성하는 코맨드를 만드는 것


class Command(BaseCommand):

    help = 'this command creates facilities'

    # def add_arguments(self, parser):
    #     parser.add_argument(
    #         '--times',
    #         help="how many times do you want to tell me that I love you?"
    #     )

    def handle(self, *args, **options):
        facilities = [
            "Private entrance",
            "Paid parking on premises",
            "Paid parking off premises",
            "Elevator",
            "Parking",
            "Gym",
        ]
        for f in facilities:
            Facility.objects.create(name=f)
        self.stdout.write(self.style.SUCCESS(
            f'{len(facilities)} facilities created'))
