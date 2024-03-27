"""Design URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from blog import views

urlpatterns = [
    path('',views.index,name='index'),
    path('team_intro',views.team_intro,name='team'),
    path('research',views.research,name='research'),
    path('future',views.future,name='future'),
    # path('',views.index1,name='index1'),
    path('admin/', admin.site.urls),
    path('blog/',include('blog.urls')),
    path('login/',views.login,name='login'),
    path('logsucess/',views.logsuccess,name='loginsuccess'),
    path('register/',views.register,name='register'),
    path('forget',views.forget_password,name='forget'),
    path('reset',views.reset,name='reset'),
    path('logout',views.log_out,name='logout'),
    path('summernote/', include('django_summernote.urls')),
    
]

