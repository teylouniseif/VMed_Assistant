from django.shortcuts import render
from django.contrib.auth.decorators import permission_required
from django.shortcuts import get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from pbskids.forms import DataFileForm, GraphFileForm
from pbskids.models import GraphFile, DataFile
from django.conf import settings
from django.template import RequestContext
from django.core.exceptions import ValidationError
from django.utils.translation import ugettext_lazy as _
from django.core.files.storage import FileSystemStorage
import numpy
from PIL import Image
import io
from django.core.files.uploadedfile import InMemoryUploadedFile
import os
import imageio
from django.core.files import File
import mimetypes
from django.http import HttpResponse
from pbskids.services.medanalyzer import extract_med_info



def treat_file(): 
    fl_path=extract_med_info()
    return download_file(fl_path)

def download_file(fl_path):
    # fill these variables with real values
    #for file in os.listdir(settings.DATA_ROOT):
    #    fl_path=os.path.join(settings.DATA_ROOT, file)
    filename = 'medical_info'

    fl = open(fl_path, 'rb')
    mime_type, _ = mimetypes.guess_type(fl_path)
    print(mime_type)
    response = HttpResponse(fl, content_type=mime_type)
    response['Content-Disposition'] = "attachment; filename=%s" % filename
    return response

def index(request):
    
    graph=GraphFile()
    doc=DataFile()

    # If this is a POST request then process the Form data
    if request.method == 'POST':

        # Create a form instance and populate it with data from the request (binding):
        form = DataFileForm(request.POST, request.FILES)

        # Check if the form is valid:
        if form.is_valid():

            newfile=form.save(commit=False)
            is_valid=form.validate(str(newfile.Input))
            
            if is_valid:
                #CLEAN OUt MEDIA AND IMG FOLDERS BEFORE SAVING, ONLY KEEPING RECENT FILE
                mediafolder =os.listdir(settings.DATA_ROOT) 
                fs = FileSystemStorage()
                for mediafile in mediafolder:
                    fs.delete(os.path.join(settings.DATA_ROOT, mediafile))
                    
                imgfolder =os.listdir(settings.IMG_ROOT)
                fs = FileSystemStorage()
                for imgfile in imgfolder:
                    fs.delete(os.path.join(settings.IMG_ROOT, imgfile))
                #save file
                form.save()
                #call subprocess to fit machine learning model on new test set
                #new graph will be saved by subprocess, also needs to call collectstatic
            else:
                pass
                
    # If this is a GET (or any other method) create the default form.
    else:
        
        if(request.GET.get('mybtn')):
            try:
                return treat_file()
            except Exception as e:
                print(e)
                pass
        
        
        form = DataFileForm()
     
    
    #DEFAULT GRAPH
    context = {
        'graph': graph,
        'form': DataFileForm(),
    }
    

    # Render the HTML template index.html with the data in the context variable
    return render(request, 'index.html', context=context)
