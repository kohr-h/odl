import os

import odl
import numpy as np

from odl.contrib.mrc import (
    FileReaderMRC, FileWriterMRC, mrc_header_from_params)


# --- Reading --- #

file_path = os.path.abspath('/home/zickert/One_particle/rna_phantom.mrc')

# File readers can be used as context managers like `open`. As argument,
# either a file stream or a file name string can be used.

with FileReaderMRC(file_path) as reader:
    print(reader.data_dtype)
    data2 = reader.read_data(swap_axes=False)
