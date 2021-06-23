from django.core.management.base import BaseCommand
from django_seed import Seed
from users.models import User

# 장고는 management/commands 라는 디렉토리의 모듈을 custom command로 인식한다.
# 어플리케이션 폴더에 생성하면 된다.


class Command(BaseCommand):

    help = 'this command creates many users'

    def add_arguments(self, parser):
        parser.add_argument(
            '--number',
            default=2,
            type=int,
            help="how many users do you want to create?"
        )

    def handle(self, *args, **options):
        number = options.get("number", 1)  # 두번째 인자: 입력이 없을때
        seeder = Seed.seeder()
        seeder.add_entity(
            User, number, {"is_staff": False, "is_superuser": False})  # 만드는 오브젝트는 이 속성을 가질 수 없음.
        seeder.execute()
        self.stdout.write(self.style.SUCCESS(f"{number} users created!"))
