from django import forms
from . import models


class SearchForm(forms.Form):

    pass

class CreateDetailForm(forms.ModelForm):
    class Meta:
        model = models.Diagnosis
        fields = [
            "file",
            #"disease",
        ]
    def save(self, pk, *args, **kwargs):
        photo = super().save(commit=False)
        patient = models.Patient.objects.get(pk=pk)
        photo.patient = patient
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
