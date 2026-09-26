import os

# Avoid OMP "libiomp5md.dll already initialized"
# when matplotlib + torch are both imported (set before torch initializes OpenMP).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
