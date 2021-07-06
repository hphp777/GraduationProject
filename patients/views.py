from django.views.generic import ListView, DetailView, View
from django.shortcuts import render
from django.core.paginator import Paginator
from . import models, forms
import patients

# Create your views here.

class PatientView(ListView):

    """PatientsView Definition"""
    template_name = "patient_list.html"
    paginate_by = 12
    paginate_orphans = 5
    ordering = "created"
    context_object_name = "patients"

class PatientDetail(DetailView):

    """ PatientDetail Definition """

    model = models.Room

class SearchView(View):
    """ SearchView Definition """

    def get(self, request):

        form = forms.SearchForm(request.GET)

        if form.is_valid():

            filter_args = {}


            return render(
                request, "search.html", {
                    "form": form, "patients": patients}
            )


        return render(request, "search.html", {"form": form})