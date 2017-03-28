from __future__ import print_function
import numpy as np
import struct
import math
import os
import re

class Tile(object):
    def __init__(self, filename):
        self.elevation = self.read_file(filename)
        self.elev_units = 'meters'
        self.dimensions = ('lat','lon')
        self.lat, self.lon = self.compute_coordinates(filename)


    @staticmethod
    def read_file(filename):
        # Documentation for SRTM files: https://dds.cr.usgs.gov/srtm/version2_1/Documentation/SRTM_Topo.pdf
        # Read the file size - this will be in bytes. Each value is two bytes, so this tells us how
        # many values there are.
        stats = os.stat(filename)
        n = math.sqrt(stats.st_size/2)
        if n % 1 > 0:
            raise ValueError('The array of read values cannot be square - this is not expected!')
        n = int(n)
        elev = np.zeros((n, n), np.float32)
        with open(filename, 'rb') as f:
            for i in range(n):
                for j in range(n):
                    val = f.read(2)
                    elev[i, j] = np.float32(struct.unpack('>h', val)[0])

        elev[elev < -30000] = np.nan
        return np.flipud(elev)

    def compute_coordinates(self, filename):
        basename = os.path.basename(filename)
        m = re.search('[NS]\d+', basename)
        latcorn = float(m.group()[1:])
        if m.group()[0] == 'S':
            latcorn *= -1

        m = re.search('[WE]\d+', basename)
        loncorn = float(m.group()[1:])
        if m.group()[0] == 'W':
            loncorn *= -1

        # Assume a 1 degree tile, the number of grid points tells us the resolution
        nlat = self.elevation.shape[0] - 1
        latres = 1.0 / nlat
        nlon = self.elevation.shape[1] - 1
        lonres = 1.0 / nlon
        lats = np.arange(latcorn, latcorn+1.0, latres)
        lats = np.concatenate([lats, np.array([latcorn+1.0])])
        lons = np.arange(loncorn, loncorn+1.0, lonres)
        lons = np.concatenate([lons, np.array([loncorn+1.0])])

        # lats order may need flipped
        return lats, lons
