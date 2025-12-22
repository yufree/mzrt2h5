import sys
import os

print(f"Python version: {sys.version}")
print(f"sys.path: {sys.path}")

# Check if there's a local numpy module
for path in sys.path:
    if os.path.exists(os.path.join(path, "numpy.py")):
        print(f"Found numpy.py at: {os.path.join(path, 'numpy.py')}")
    if os.path.exists(os.path.join(path, "numpy")):
        print(f"Found numpy directory at: {os.path.join(path, 'numpy')}")
        if os.path.isdir(os.path.join(path, "numpy")):
            print(f"Directory contents: {os.listdir(os.path.join(path, 'numpy'))[:5]}...")

# Try importing numpy
try:
    import numpy
    print(f"numpy imported successfully")
    print(f"numpy.__file__: {getattr(numpy, '__file__', 'Not found')}")
    print(f"numpy.__version__: {getattr(numpy, '__version__', 'Not found')}")
    print(f"numpy attributes: {[attr for attr in dir(numpy) if not attr.startswith('_')][:10]}...")
except Exception as e:
    print(f"Error importing numpy: {e}")
