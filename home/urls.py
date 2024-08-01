from django.urls import path,include
from . import views

app_name = 'home'

urlpatterns = [
    path('',views.index,name='index'),
    path('user_login/', views.user_login, name='user_login'),
    path('dashboard/',views.dashboard,name='dashboard'),
    path('logout/',views.logout,name='logout'),
    path("password_reset/", views.password_reset_request, name="password_reset"),
    path('register/',views.register,name='register'),
    path('search/',views.search,name='search'),
    path('instant_exam/',views.instant_exam, name='instant_exam'),
    path('vitals/',views.vitals, name='vitals'),
    path('symptoms/',views.symptoms, name='symptoms'),
    path('directory/',views.directory,name='directory'),
    path('display/',views.display,name='display'),
    path('survey/',views.survey,name='survey'),
    path('raise_query/',views.raise_query,name='raise_query'),
    path('profile/',views.profile,name='profile'),
    path('medical/',views.medical,name='medical'),
    
    
 

]
