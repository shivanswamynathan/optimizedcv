import os
from django.core.asgi import get_asgi_application
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'resume_enhancer.settings')

application = get_asgi_application()