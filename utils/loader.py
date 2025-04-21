from pygt3x.reader import FileReader
import pandas as pd

def load_gt3x(path):
    with FileReader(path) as reader:
        dfraw = reader.to_pandas()
        dfraw['Datetime'] = pd.to_datetime(dfraw.index, unit='s')
        dfraw['vector_magnitude'] = (dfraw['X']**2 + dfraw['Y']**2 + dfraw['Z']**2)**0.5
        return dfraw


def load_raw_accel_file(path):
    return load_gt3x(path)