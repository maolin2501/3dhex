"""Independent vertex-only jobs; never overwrite the user's previous outputs."""
from pathlib import Path
from .aligned_regression import main

if __name__=='__main__':
    main('vertex','5592ea9c54635f97836ba6a0',Path('reference_meshes/shoulder_v1/vertex'))
