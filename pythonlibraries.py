try:
    import numpy
    print("NumPy is installed, version:", numpy.__version__)
except ImportError:
    print("NumPy is not installed.")

try:
    import pandas
    print("Pandas is installed, version:", pandas.__version__)
except ImportError:
    print("Pandas is not installed.")

try:
    import matplotlib
    print("Matplotlib is installed, version:", matplotlib.__version__)
except ImportError:
    print("Matplotlib is not installed.")

try:
    import seaborn
    print("Seaborn is installed, version:", seaborn.__version__)
except ImportError:
    print("Seaborn is not installed.")

try:
    import sklearn
    print("Scikit-learn is installed, version:", sklearn.__version__)
except ImportError:
    print("Scikit-learn is not installed.")