from django.http import Http404
from django.views.generic import ListView, DetailView, View, UpdateView, FormView
from django.shortcuts import render, redirect, reverse
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.messages.views import SuccessMessageMixin
# from users import mixins as user_mixins
from . import models, forms

# Create your views here.

class PatientView(ListView):

    """PatientsView Definition"""
    
    model=models.Patient
    paginate_by = 12
    paginate_orphans = 5
    ordering = "created"
    context_object_name = "patients"

class PatientDetail(DetailView):
    
    """ RoomDetail Definition """

    model = models.Patient

class AllocationView(ListView):

    """AllocationView Definition"""
    

class RegistrationView(FormView):

    """RegisterationView Defenition"""

    form_class = forms.CreatePatientForm
    template_name = "patients/patient_registration.html"

    def form_valid(self, form):
        patient = form.save()
        patient.doctor = self.request.user
        patient.save()
        form.save_m2m()
        messages.success(self.request, "Patient Registration")
        return redirect(reverse("patients:detail", kwargs={"pk": patient.pk}))

class DiagnosisView():
    
    """ResisterView Defenition"""    

class SearchView(View):

    """ SearchView Definition """

    def get(self, request):

        id = request.GET.get("id")

        form = forms.SearchForm(request.GET)

        if form.is_valid():

            images = form.cleaned_data.get("image")
            disease1 = form.cleaned_data.get("disease1")
            disease2 = form.cleaned_data.get("disease2")
            disease3 = form.cleaned_data.get("disease3")

            filter_args = {}

            if id != "Anyone":
                    filter_args["id"] = id

            for image in images:
                    filter_args["images"] = image

            patients = models.Patient.objects.filter(**filter_args)

            return render(
                request, "search.html", {
                    "form": form, "patients": patients}
            )
        else:
    
            form = forms.SearchForm()

        return render(request, "search.html", {"form": form})