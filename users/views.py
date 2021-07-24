from django.views.generic import FormView
from django.urls import reverse_lazy
from django.shortcuts import redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.messages.views import SuccessMessageMixin
from django.contrib import auth
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect

from .forms import UserForm
# Create your views here.


def login(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return HttpResponseRedirect('/patients/list/')
        else:
            return render(request, 'login.html', {'error':'username or password is incorrect'})
            # 로그인 실패시 'username or password is incorrect' 메시지를 띄움  
    else:
        return render(request, 'login.html')

def log_out(request):
    logout(request)
    return redirect(reverse_lazy("core:home"))


def signup(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect(reverse_lazy("core:home"))
    else:
        form = UserForm()
    return render(request, 'signup.html', {'form': form})