from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'multi_object_throwing'

def get_data_files():
    data_files = [
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]
    
    # Include data directories
    data_dirs = ['config', 'description', 'brt_data', 'hedgehog_data', 'hedgehog_revised', 'launch']
    for data_dir in data_dirs:
        for root, dirs, files in os.walk(data_dir):
            if files:
                dest = os.path.join('share', package_name, root)
                src = [os.path.join(root, f) for f in files]
                data_files.append((dest, src))
    return data_files

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=get_data_files(),
    install_requires=[
        'numpy>=1.19.0',
        'rospkg>=1.4.0',
        'scipy>=1.7.0',
    ],
    scripts=[
        'scripts/trajectory_tracking.py',
        'scripts/hedgehog.py',
        'scripts/NN_search.py',
        'scripts/RL_throwing.py',
        'scripts/sample_generation.py',
        'scripts/training.py',
        'scripts/trajectory_generator.py',
    ],
    python_requires='>=3.6',
)
