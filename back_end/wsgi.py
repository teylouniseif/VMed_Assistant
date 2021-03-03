"""
WSGI config for back_end project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os
import sys
#Add the app's directory to the python path
sys.path.append('/var/www/html2')
sys.path.append('/var/www/html2/back_end')

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'back_end.settings')

application = get_wsgi_application()
