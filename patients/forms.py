from django import forms
from . import models
from .diagnosis import *


class SearchForm(forms.Form):

    pass


class CreateDetailForm(forms.ModelForm):
    class Meta:
        model = models.Image
        fields = [
            "file",
        ]

    def save(self, pk, *args, **kwargs):
        photo = super().save(commit=False)
        patient = models.Patient.objects.get(pk=pk)
        photo.patient = patient
        photo.save()
        photo = models.Image.objects.get(id=photo.id)
        disease, percentage = diagnosis(
            model, photo.filename(), "uploads/patient_images/", train_df_main, labels
        )
        detection(photo.filename())
        photo.detection_file = "patient_detect_images/exp/" + photo.filename()
        # photo.cam=gradcam(photo.filename(), "uploads/patient_images/")
        photo.disease1 = disease[0]
        photo.disease2 = disease[1]
        photo.disease3 = disease[2]
        photo.percentage1 = percentage[0]
        photo.percentage2 = percentage[1]
        photo.percentage3 = percentage[2]
        photo.save()


class CreatePatientForm(forms.ModelForm):
    class Meta:
        model = models.Patient
        fields = [
            "name",
            "id",
            "age",
            "gender",
            "doctor",
            "description",
        ]
