import os
import subprocess
import importlib.util
import pkg_resources
import sys

# List of default packages that come with Python
default_packages = []
for name in sys.modules:
    if name.startswith('__'):
        default_packages.append(name[2:])
    else:
        default_packages.append(name)

for name in os.listdir(os.path.dirname(os.__file__)):
    if name.startswith('__'):
        default_packages.append(name[2:])
    elif name.endswith('.py'):
        default_packages.append(name[:-3])
    else:
        default_packages.append(name)

def is_package_importable(package):
    """
    Check if the given package can be imported.
    """
    try:
        __import__(package)
        return True
    except:
        return False

# Define the environment name and Python version
env_name = 'env_ai'
python_version = '3.11.5'

# Directory containing Python files with dependencies
directory_path = r'C:/Users/gorth/Dropbox (MIT)/Shared/Papers/AI for nuclear/ai-nuclear-nn_test_long_run'

# Activate the conda environment
conda_cmd = f'conda activate {env_name}'
subprocess.run(conda_cmd.split(), shell=True)
subprocess.run('pip install --upgrade setuptools -y', shell=True)

# Find all .py files in the specified directory and its subdirectories
py_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory_path) for f in filenames if f.endswith('.py')]

# Scan the .py files for import statements and extract the required packages
required_packages = set()
for py_file in py_files:
    with open(py_file, 'r') as f:
        for line in f:
            if line.startswith('import '):
                package = line.split()[1]
                package_name = package.split('.')[0]

                # Skip empty or invalid package names
                if not package_name or any(char in package_name for char in [' ', ',', ':', ';', '(', ')']):
                    continue

                # Check if the package is already imported
                if (package_name in locals() or
                    package_name in globals() or
                    os.path.basename(package_name + '.py') in os.listdir(directory_path) or
                    package_name in default_packages):
                    continue
                required_packages.add(package_name)

required_packages = [item if item != 'sklearn' else 'scikit-learn' for item in required_packages]

print("Required packages:" + str(required_packages))
# Install the required packages
for package in required_packages:
    installed = is_package_importable(package)
    if not installed:
        print(f'Installing {package}...')
        try:
            subprocess.run(f'conda install -y {package}', shell=True)
            installed = is_package_importable(package)
        except ModuleNotFoundError:
            print(f'Could not install {package} with conda')
        if not installed:
            try:
                subprocess.run(f'conda install -c conda-forge -y {package}', shell=True)
                installed = is_package_importable(package)
            except ModuleNotFoundError:
                print(f'Could not install {package} with conda-forge')
        if not installed:
            try:
                subprocess.run(f'pip install -Uq {package}', shell=True)
                installed = is_package_importable(package)
            except ModuleNotFoundError:
                print(f'Could not install {package}')
        if not installed:
            print(f'Could not install {package}')
