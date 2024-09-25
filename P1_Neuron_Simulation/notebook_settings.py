from IPython import get_ipython
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))


ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")