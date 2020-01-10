"""braintumor URL Configuration

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
from django.urls import path
from brain.views import *
from django.conf import settings # new
from django.urls import path, include # new
from django.conf.urls.static import static
from rest_framework.documentation import include_docs_urls
from rest_framework_swagger.views import get_swagger_view

schema_view = get_swagger_view(title='Raseedi API')

urlpatterns = [
    path('swagdoc/', schema_view),
    path('create_sets/', CreateSets.as_view()),
    path('admin/', admin.site.urls),
    path('create_patient/', CreatePatient.as_view()),
    path('create_scans/', CreateScans.as_view()),
    path('docs', include_docs_urls(title="Raseedi Restful Api")),

]

if settings.DEBUG: # new
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)