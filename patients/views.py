from django.views.generic import ListView, DetailView, View
from django.core.paginator import Paginator
from django.shortcuts import render
from . import models, forms


# Create your views here.

class PatientView(ListView):

    """PatientsView Definition"""
    
    # template_name = "patient_list.html"
    model=models.Patient
    paginate_by = 12
    paginate_orphans = 5
    ordering = "created"
    context_object_name = "patients"

class PatientDetail(DetailView):
    
    """ RoomDetail Definition """

    model = models.Patient

class AllocationView():

    """AllocationView Definition"""

class ResisterView():

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