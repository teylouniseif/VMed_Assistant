from django import forms
from django.conf import settings
import pandas as pd
from pbskids.models import DataFile 
from pbskids.models import GraphFile
import os
import numpy as np

class DataFileForm(forms.ModelForm):
    
    class Meta:
        model = DataFile
        fields = ('Input',)
    
    def validate(self, filename):
        
        #return true by default
        return True
        
        dtypes = {'event_id': 'object', 'game_session': 'object', 'timestamp': 'object',\
                    'event_data': 'object', 'installation_id':'object', 'event_count':'int64',\
                    'event_code':'int64', 'game_time':'int64', 'title':'object', 'type':'object',\
                    'world':'object'}
        
        #try reading the file with pandas, if error file is not valid
        print(os.path.join(settings.DATA_ROOT, filename))
        
        try:
            df = pd.read_csv(os.path.join(settings.DATA_ROOT, filename))
        except:
            return False

        # Check if csv file is formatted properly
        if dict(df.dtypes) != dtypes:
            return False
            
        #file is valid
        return True
    
class GraphFileForm(forms.ModelForm):
    
    class Meta:
        model = GraphFile
        fields = ('image',)
            
            
