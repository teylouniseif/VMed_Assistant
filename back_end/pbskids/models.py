from django.db import models
from django.contrib.auth.models import User
import csv
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.urls import reverse


# Create your models here.

class GraphFile(models.Model):
    image = models.FileField(upload_to=settings.IMG_URL,  default=None)
    
class DataFile(models.Model):
    Input = models.FileField(upload_to=settings.DATA_URL,  default=None)
    uploaded_at = models.DateTimeField(auto_now_add=True)
