from django.db import models
from django.urls import reverse
from core import models as core_models
from users import models as user_models

# Create your models here.
class Patient(core_models.TimeStampedModel):

    """Patients Model Dedinition"""

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
        (EFFUSION , "effusion"),
        (EMPHYSEMA, "emphysema"),
        (FIBROSIS, "fibrosis"),
        (HERNIA, "hernia"),
        (INFILTRATION, "infiltration"),
        (MASS, "mass"),
        (NODULE, "nodule"),
        (PLEURAL_THICKENING, "pleural_thickening"),
        (PNEUMONIA, "pneumonia"),
        (PNEUMOTHORAX, "pneumothorax")
    )

    name = models.CharField(max_length=140)
    description = models.TextField()
    disease1 = models.CharField(
        choices=DISEASE_CHOICES, blank=False, max_length= 20, default=NO_FINDING
    )
    disease2 = models.CharField(
        choices=DISEASE_CHOICES, blank=False, max_length= 20, default=NO_FINDING
    )
    disease3 = models.CharField(
        choices=DISEASE_CHOICES, blank=False, max_length= 20, default=NO_FINDING
    )
    avatar = models.ImageField(upload_to="avatars", null=True, blank=True)
    doctor = models.ForeignKey(user_models.User, related_name="patients", on_delete=models.CASCADE)