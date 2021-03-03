"""back_end URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from django.urls import include, re_path

urlpatterns = [
    path('admin/', admin.site.urls),
]

urlpatterns += [
    #path('pbskids/', include('pbskids.urls')),
    path('', include('pbskids.urls')),
]

#Add URL maps to redirect the base URL to our application
from django.views.generic import RedirectView
#urlpatterns += [
#    path('', RedirectView.as_view(url='pbskids/', permanent=True)),
#]

# Use static() to add url mapping to serve static files during development (only)
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.views.static import serve

urlpatterns += staticfiles_urlpatterns()
urlpatterns += [
        re_path(r'^img/(?P<path>.*)$', serve, {
            'document_root': settings.IMG_ROOT,
        }),
    ]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.IMG_URL, document_root=settings.IMG_ROOT)
    urlpatterns += static(settings.DATA_URL, document_root=settings.DATA_ROOT)
