from django.http import Http404
from django.views.generic import ListView, DetailView, View, UpdateView, FormView
from django.shortcuts import render, redirect, reverse
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.messages.views import SuccessMessageMixin
from django.core.paginator import Paginator, EmptyPage
# from users import mixins as user_mixins
from . import models, forms

# Create your views here.

def all_patient(request):
    page = request.GET.get("page", 1)
    patient_list = models.Patient.objects.all()
    paginator = Paginator(patient_list, 10, orphans=5)
    print("patients: " , patient_list)
    try:
        patients = paginator.page(int(page))
        return render(request, "patients/patient_list.html", {"patients": patients})
    except EmptyPage:
        return redirect("/patients/list")

class PatientDetail(DetailView):
    
    """ RoomDetail Definition """

    model = models.Patient

class AllocationView(ListView):

    """AllocationView Definition"""
    
def registrate(request):
    if request.method=='POST':
        form = forms.CreatePatientForm(request.POST)
        if form.is_valid():
            post = form.save()
            post.save()
        return redirect('/patiensts/list')
    else: #GET
        form = forms.CreatePatientForm()
    return render(request, 'patients/patient_registration.html', {'form': form})

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