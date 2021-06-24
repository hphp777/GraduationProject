from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.


def all_rooms(request):
    return HttpResponse(request, "room_list.html")
