from django.views.generic import FormView, DetailView
from django.urls import reverse_lazy
from django.shortcuts import redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.messages.views import SuccessMessageMixin
from django.contrib import auth
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from . import models

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
    messages.info(request, f"See you later")
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

class UserProfileView(DetailView):
    
    model = models.User

    # template안에 더 많은 context를 사용할 수 있게 해 줄것이다.
    # super를 의무적으로 호출해야 하는데 이유는 user_obj를 기본으로 주기 때문에
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["hello"] = "Hello!"
        return context