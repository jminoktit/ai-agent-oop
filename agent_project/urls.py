"""
URL configuration for agent_project project.
"""
import os
from django.contrib import admin
from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse


def serve_react_app(request, path=''):
    """Serve the React frontend SPA."""
    index_path = os.path.join(settings.REACT_BUILD_DIR, 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return HttpResponse(f.read(), content_type='text/html')
    from django.shortcuts import render
    return render(request, 'agent_app/index.html')


# API endpoints (agent_app) - these handle both HTML and JSON
api_patterns = [
    path('', include('agent_app.urls')),
]

urlpatterns = [
    path('admin/', admin.site.urls),
    # Agent app API - serves both HTML (templates) and JSON (React)
    path('', include('agent_app.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    from django.views.static import serve as static_serve
    dist_dir = settings.REACT_BUILD_DIR
    assets_dir = dist_dir / 'assets'
    if os.path.exists(assets_dir):
        urlpatterns += [
            path('assets/<path:path>', static_serve, {'document_root': str(assets_dir)}),
        ]
