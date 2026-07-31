#!/usr/bin/env python
"""Copy React build to Django static directory for production serving."""
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REACT_DIST = os.path.join(BASE_DIR, 'frontend', 'dist')
DJANGO_STATIC = os.path.join(BASE_DIR, 'staticfiles', 'frontend')

def main():
    if not os.path.exists(REACT_DIST):
        print("React build not found. Run 'cd frontend && npm run build' first.")
        return

    # Clean destination
    if os.path.exists(DJANGO_STATIC):
        shutil.rmtree(DJANGO_STATIC)

    # Copy dist to staticfiles/frontend
    shutil.copytree(REACT_DIST, DJANGO_STATIC)
    print(f"Copied React build to {DJANGO_STATIC}")

    # List files
    for root, dirs, files in os.walk(DJANGO_STATIC):
        level = root.replace(DJANGO_STATIC, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')

if __name__ == '__main__':
    main()
