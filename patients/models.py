from django.db import models
from django.urls import reverse
from core import models as core_models
import os

class Image(core_models.TimeStampedModel):

    """Image Model Definition"""

    ATELECTASIS = "atelectasis"
    CARDIOMEGALY = "cardiomegaly"
    CONSOLIDATION = "consolidation"
    EDEMA = "edema"
    EFFUSION = "effusion"
    EMPHYSEMA = "emphysema"
    FIBROSIS = "fibrosis"
    HERNIA = "hernia"
    INFILTRATION = "infiltration"
    MASS = "mass"
    NODULE = "nodule"
    PLEURAL_THICKENING = "pleural_thickening"
    PNEUMONIA = "pneumonia"
    PNEUMOTHORAX = "pneumothorax"
    NO_FINDING = "no_finding"

    DISEASE_CHOICES = (
        (NO_FINDING, "no_finding"),
        (ATELECTASIS, "atelectasis"),
        (CARDIOMEGALY, "cardiomegaly"),
        (CONSOLIDATION, "consolidation"),
        (EDEMA, "edema"),
        (EFFUSION, "effusion"),
        (EMPHYSEMA, "emphysema"),
        (FIBROSIS, "fibrosis"),
        (HERNIA, "hernia"),
        (INFILTRATION, "infiltration"),
        (MASS, "mass"),
        (NODULE, "nodule"),
        (PLEURAL_THICKENING, "pleural_thickening"),
        (PNEUMONIA, "pneumonia"),
        (PNEUMOTHORAX, "pneumothorax"),
    )

    LOW = "low"
    MIDDLE = "middle"
    HIGH = "high"

    SERIOUSNESS_CHOICES = ((LOW, "low"), (MIDDLE, "middle"), (HIGH, "high"))

    file = models.ImageField(upload_to="patient_images")
    patient = models.ForeignKey(
        "Patient", related_name="images", on_delete=models.CASCADE
    )
    disease1 = models.CharField(
        choices=DISEASE_CHOICES, blank=False, max_length=20, default=NO_FINDING
    )
    disease2 = models.CharField(
        choices=DISEASE_CHOICES, blank=False, max_length=20, default=NO_FINDING
    )
    disease3 = models.CharField(
        choices=DISEASE_CHOICES, blank=False, max_length=20, default=NO_FINDING
    )
    seriousness = models.CharField(
        choices=SERIOUSNESS_CHOICES, blank=False, max_length=10, default=LOW
    )

    def __str__(self):
        return os.path.basename(self.file.name)


class Area(core_models.TimeStampedModel):

    """Place Model Definition"""

    CPR = "cpr"
    CRITICAL_CARE_AREAS = "critical_care_areas"
    CASUALITY_DEPARTMENT = "casualty_department "
    EMERGENCY_PATIENT_AREAS = "emergency_patient_areas"

    AREA_CHOICES = (
        (CPR, "cpr"),
        (CRITICAL_CARE_AREAS, "critical_care_areas"),
        (CASUALITY_DEPARTMENT, "casualty_department"),
        (EMERGENCY_PATIENT_AREAS, "emergency_patient_areas"),
    )

    # patient2 = models.ForeignKey(
    #     "Patient2", related_name="images", on_delete=models.CASCADE
    # )
    area = models.CharField(
        choices=AREA_CHOICES, blank=False, max_length=10, default=CASUALITY_DEPARTMENT
    )

    # def __str__(self):
    #     return self.patient.name


class Patient(core_models.TimeStampedModel):

    """Patient Model Definition"""

    FEMALE = "female"
    MALE = "male"

    GENDER_CHOICES = ((FEMALE, "female"), (MALE, "male"))

    name = models.CharField(max_length=140, null=True, blank=True)
    id = models.IntegerField(primary_key=True)
    age = models.IntegerField()
    gender = models.CharField(
        choices=GENDER_CHOICES, blank=False, max_length=10, default=FEMALE
    )
    doctor = models.ForeignKey(
        "users.User", related_name="patients", on_delete=models.CASCADE
    )
    # avatar = models.ImageField(upload_to="avatars", null=True, blank=True)

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse("patients:detail", kwargs={"pk": self.pk})

    def first_image(self):
        image, = self.images.all()[:1]
        return image.file.url
