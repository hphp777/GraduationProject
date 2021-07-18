from django.db import models
from django.urls import reverse
from core import models as core_models
import os

class Image(core_models.TimeStampedModel):

    """ Image Model Definition """

    file = models.ImageField(upload_to="patient_images")
    patient = models.ForeignKey("Patient", related_name="images", on_delete=models.CASCADE)
    def __str__(self):
        return os.path.basename(self.file.name)


class Patient(core_models.TimeStampedModel):

    """Patient Model Definition"""

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
    description = models.TextField(default="", blank=True)
    # images=models.ManyToManyField("Image",related_name="patients",blank=True)
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
    area = models.CharField(
        choices=AREA_CHOICES, blank=False, max_length=30, default=CASUALITY_DEPARTMENT
    )

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse("patients:detail", kwargs={"pk": self.pk})

    def get_image(self):
        try:
            image = self.images.all()[:1]
            return image   
        except ValueError:
            return None