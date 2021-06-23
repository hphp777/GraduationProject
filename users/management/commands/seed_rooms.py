from django.core.management.base import BaseCommand
from django_seed import Seed
from django.contrib.admin.utils import flatten
import random
from rooms import models as room_models
import rooms
from users import models as user_models
# 장고는 management/commands 라는 디렉토리의 모듈을 custom command로 인식한다.
# 어플리케이션 폴더에 생성하면 된다.


class Command(BaseCommand):

    help = 'this command creates rooms'

    def add_arguments(self, parser):
        parser.add_argument(
            '--number',
            default=2,
            type=int,
            help="how many rooms do you want to create?"
        )

    def handle(self, *args, **options):
        number = options.get("number")
        seeder = Seed.seeder()
        all_users = user_models.User.objects.all()
        room_types = room_models.RoomType.objects.all()
        seeder.add_entity(room_models.Room, number, {  # 만드는 오브젝트에 대한 설정
            'name': lambda x: seeder.faker.address(),
            'host': lambda x: random.choice(all_users),
            'room_type': lambda x: random.choice(room_types),
            'guests': lambda x: random.randint(1, 20),
            'price': lambda x: random.randint(0, 300),
            'bed': lambda x: random.randint(1, 5),
            'bedrooms': lambda x: random.randint(1, 5),
            'baths': lambda x: random.randint(1, 5),
        })
        created_photos = seeder.execute()
        # shape을 예쁘게 변경
        created_clean = flatten(list(created_photos.values()))
        amenities = room_models.Amenity.objects.all()
        facilities = room_models.Facility.objects.all()
        rules = room_models.HouseRule.objects.all()
        for pk in created_clean:
            # primary key로 room을 찾음
            room = room_models.Room.objects.get(pk=pk)
            for i in range(3, random.randint(10, 30)):
                room_models.Photo.objects.create(
                    caption=seeder.faker.sentence(),
                    rooms=room,
                    file=f"room_photos/{random.randint(1,31)}.webp"
                )
            # many to many field 추가하는 법
            for a in amenities:
                magic_number = random.randint(0, 15)
                if magic_number % 2 == 0:
                    room.Amenities.add(a)
            for f in facilities:
                magic_number = random.randint(0, 15)
                if magic_number % 2 == 0:
                    room.Facilities.add(f)
            for r in rules:
                magic_number = random.randint(0, 15)
                if magic_number % 2 == 0:
                    room.house_rules.add(r)

        self.stdout.write(self.style.SUCCESS(f"{number} rooms created!"))
