import os

def find_urdf_objects():
    """Return a {name: path} dict of .urdf objects in this module"""
    assets_dir = os.path.dirname(__file__)
    urdf_files = [f for f in os.listdir(assets_dir) if f.endswith('.urdf')]
    return {
        f.split('.')[0]: os.path.join(assets_dir, f) for f in urdf_files
    }
