from django.views.generic import FormView, DetailView, UpdateView
from django.urls import reverse_lazy
from django.shortcuts import redirect, reverse
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from . import forms
from . import models

# Create your views here.


class LoginView(FormView):
    template_name = "users/login.html"
    form_class = forms.LoginForm
    success_url = reverse_lazy("core:home")

    def form_valid(self, form):
        email = form.cleaned_data.get("email")
        password = form.cleaned_data.get("password")
        user = authenticate(self.request, username=email, password=password) # 기본적으로 로그인을 하려면 username, password가 필요
        if user is not None:
            login(self.request, user)
        return super().form_valid(form)

def log_out(request):
    messages.info(request, f"See you later")
    logout(request)
    return redirect(reverse("core:home"))


class SignUpView(FormView):

    template_name = "users/signup.html"
    form_class = forms.SignUpForm
    success_url = reverse_lazy("core:home")

    def form_valid(self, form):
        form.save()
        email = form.cleaned_data.get("email")
        password = form.cleaned_data.get("password")
        user = authenticate(self.request, username=email, password=password)
        if user is not None:
            login(self.request, user)
        return super().form_valid(form)

class UserProfileView(DetailView):
    
    context_object_name = "user_obj"
    model = models.User

    # template안에 더 많은 context를 사용할 수 있게 해 줄것이다.
    # super를 의무적으로 호출해야 하는데 이유는 user_obj를 기본으로 주기 때문에
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["hello"] = "Hello!"
        return context
        
class UpdateUserView(UpdateView):
    
    model = models.User
    template_name = "users/update-profile.html"
    fields = (
        "first_name",
        "last_name",
        "avatar",
        "gender",
        "bio",
        "birthdate",
        "language",
        "currency",
    )

    # 수정하기를 원하는 객체를 반환해 줄 것이다.
    def get_object(self, queryset=None):
        return self.request.user