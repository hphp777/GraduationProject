from django.core.management.base import BaseCommand
from rooms.models import Amenity

# 장고는 management/commands 라는 디렉토리의 모듈을 custom command로 인식한다.
# 어플리케이션 폴더에 생성하면 된다.


class Command(BaseCommand):

    help = 'this command says that loves me'

    # def add_arguments(self, parser):
    #     parser.add_argument(
    #         '--times',
    #         help="how many times do you want to tell me that I love you?"
    #     )

    def handle(self, *args, **options):
        amenities = [
            "Air conditioning",
            "Alarm Clock",
            "Balcony",
            "Bathtub",
            "Bed Linen",
            "Boating",
            "Cable TV",
            "Carbon monoxide detectors",
            "Chairs",
            "Children Area",
            "Coffee Maker in Room",
            "Cooking hob",
            "Cookware & Kitchen Utensils",
            "Dishwasher",
            "Double bed",
            "En suite bathroom",
            "Free Parking",
            "Free Wireless Internet",
            "Freezer",
            "Fridge / Freezer",
            "Golf",
            "Hair Drier",
            "Heating",
            "Hot tub",
            "Indoor Pool",
            "Ironing Board",
            "Microwave",
            "Outdoor Pool",
            "Oven",
            "Queen size bed",
            "Restaurant",
            "Shopping Mall",
            "Shower",
            "Smoke detectors",
            "Sofa",
            "Stereo",
            "Swimming Pool",
            "Toilet",
            "Towels",
            "TV",
        ]
        for a in amenities:
            Amenity.objects.create(name=a)
        self.stdout.write(self.style.SUCCESS("Amenity created!"))
