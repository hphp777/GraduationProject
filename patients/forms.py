from django import forms
from . import models
from .diagnosis import *


class SearchForm(forms.Form):

    pass

class CreateDetailForm(forms.ModelForm):
    class Meta:
        model = models.Diagnosis
        fields = [
            "file",
        ]
    def save(self, pk, *args, **kwargs):
        photo = super().save(commit=False)
        patient = models.Patient.objects.get(pk=pk)
        disease,percentage=diagnosis(model,photo.filename(),"uploads/patient_images/", train_df_main, labels)
        photo.patient = patient
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
