from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma
import struct
import math
import os
import re
import warnings


class Tile(object):
    verbosity = 0

    def __init__(self, filename):
        self.file = filename
        self.elevation = self.read_file(filename)
        self.elev_units = 'meters'
        self.dimensions = ('lat', 'lon')
        self.lat, self.lon, self.latcorn, self.loncorn = self.compute_coordinates(filename)


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
        elev = ma.zeros((n, n), np.float32)
        with open(filename, 'rb') as f:
            for i in range(n):
                for j in range(n):
                    val = f.read(2)
                    elev[i, j] = np.float32(struct.unpack('>h', val)[0])

        elev[elev < -30000] = ma.masked
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
        return lats, lons, latcorn, loncorn


class TileCollection(object):
    fatal_if_tile_filled = False
    verbosity = 0

    def __init__(self, *tiles):
        self.indices = np.zeros((180, 360), np.int16)
        self.indices.fill(-1)
        self.tiles = []
        self.num_tiles = 0

        self._fill = np.nan

        for t in tiles:
            if not isinstance(t, Tile):
                raise TypeError('All inputs to the TileCollection init method must be instances of srtm.Tile')
            self.add_tile(t)

    @property
    def fill(self):
        return self._fill

    @fill.setter
    def fill(self, value):
        if isinstance(value, int):
            self._fill = float(value)
        elif isinstance(value, float):
            self._fill = value
        else:
            raise TypeError('TileCollection.fill must be a float or int (ints will be converted to floats internally)')

    @staticmethod
    def latlon2ind(lat, lon):
        i = int(lat) + 90
        j = int(lon) + 180
        return i, j

    @staticmethod
    def ind2latlon(latind, lonind):
        lat = float(latind) - 90.0
        lon = float(lonind) - 180.0
        return lat, lon

    def add_tile(self, tile):
        i, j = TileCollection.latlon2ind(tile.latcorn, tile.loncorn)
        if len(self.tiles) > 0:
            n = self.tiles[0].elevation.shape[0]
            n2 = tile.elevation.shape[0]
            if n != n2:
                raise RuntimeError('All tiles must have the same dimensions for elevation\n'
                                   '(existing = {0}x{0}, new = {1}x{1})'.format(n, n2))

        if self.indices[i, j] >= 0:
            msg = 'Tile for lon = {0}, lat = {1} already filled'.format(tile.loncorn, tile.latcorn)
            if self.fatal_if_tile_filled:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)

        self.indices[i, j] = len(self.tiles)
        self.tiles.append(tile)
        self.num_tiles += 1
        if TileCollection.verbosity > 0:
            print('Added tile for file {0} to [{1}, {2}]'.format(tile.file, i, j))

    def stitch_all_tiles(self):
        """
        Combines all stored tiles into a single masked array of elevation data
        :return: A numpy masked array. Masked areas are missing data, which occur when trying to get an area that none
        of the added tiles cover. For this method, that occurs when missing areas are needed to create a rectangular
        array
        """
        inds = np.where(self.indices >= 0)
        latmin = min(inds[0])
        latmax = max(inds[0])
        lonmin = min(inds[1])
        lonmax = max(inds[1])
        elevs, lats, lons = self._stitch_by_indices(latmin, latmax, lonmin, lonmax)
        return elevs, lats, lons

    def _stitch_by_indices(self, startlat, endlat, startlon, endlon):
        """
        Create an array of elevations from the all of or a subset of the instances of srtm.Tile that have been added to
        this collection. This is mainly an internal method, to be called by other stitching methods to get a starting
        array of elevations which can then be cut down more easily.
        :param startlat: the first tile index (in self.indices) in the latitudinal direction
        :param endlat: the last tile index in the latitudinal direction
        :param startlon: the first tile index in the longitudinal direction
        :param endlon: the last tile index in the longitudinal direction
        :return: the stitched together elevation, the corresponding latitudes, and longitudes
        """

        #TODO: check that the indended behavior occurs if given indices that don't match the added tiles

        if endlat < startlat:
            raise ValueError('endlat must be greater than startlat')
        if endlon < startlon:
            raise ValueError('endlon must be greater than startlon')

        xs = startlat
        ys = startlon
        xe = endlat + 1
        ye = endlon + 1

        n = self.tiles[0].elevation.shape[0]
        nlat = (n-1)*(xe - xs)+1
        nlon = (n-1)*(ye - ys)+1

        # Check that the size is < 2 GB, just in case
        # Assuming 4 byte floats
        var_size = 4 * nlat * nlon / 1e9
        if var_size > 2.0:
            warnings.warn('Collection array size will exceed 2 GB ({0} GB)'.format(var_size))

        elev = ma.zeros((nlat, nlon), np.float32)
        elev.fill(self.fill)
        for i in range(xs, xe):
            for j in range(ys, ye):
                # Indicies for the elev array. If we are not at the end, remove the last row and column since adjacent
                # tiles duplicate the bordered row
                ni = n - 1 if i < xe - 1 else n
                nj = n - 1 if j < ye - 1 else n

                elxs = (i - xs)*(n-1)
                elxe = (i + 1 - xs)*ni
                elys = (j - ys)*(n-1)
                elye = (j + 1 - ys)*nj

                tile_ind = self.indices[i, j]
                elev[elxs:elxe, elys:elye] = self.tiles[tile_ind].elevation[:ni, :nj]

        # Cheat to figure out the longitudes and latitudes
        all_lat = np.sort(np.unique(np.concatenate([t.lat for t in self.tiles])))
        all_lon = np.sort(np.unique(np.concatenate([t.lon for t in self.tiles])))

        if len(all_lat) != elev.shape[0]:
            raise ValueError('Different number of stitched lats than elevations in the latitudinal dimensions')
        if len(all_lon) != elev.shape[1]:
            raise ValueError('Different number of stitched lons than elevations in the longiitudinal dimensions')

        # Mask fill values
        if np.isnan(self.fill):
            elev[np.isnan(elev)] = ma.masked
        elif np.isinf(self.fill):
            elev[np.isinf(elev)] = ma.masked
        elif isinstance(self.fill, float):
            elev[elev == self.fill] = ma.masked
        else:
            raise RuntimeError('Masking for fills == {0} (type {1}) not supported'.format(self.fill,
                                                                                          type(self.fill).__name__))

        return elev, all_lat, all_lon

    def save_as_heightmap(self, filename, elev=None, limits='local'):
        if elev is None:
            elev, _, _ = self.stitch_all_tiles()
        elif not isinstance(elev, ma.core.MaskedArray) and not isinstance(elev, np.ndarray):
            raise TypeError('elev (if given) must be an instance of numpy.ma.core.MaskedArray or numpy.ndarray')

        allowed_lim_vals = ('local', 'global')
        if isinstance(limits, str):
            if limits not in allowed_lim_vals:
                raise ValueError('If given as a string, "limits" must be one of {0}'.
                                 format(', '.join(allowed_lim_vals)))
            if limits == 'local':
                elev_lim = (np.min(elev), np.max(elev))
            elif limits == 'global':
                # According to Google, the lowest elevation on earth is the lakebed of Lake Baikal at -1190 meters
                # while Everest is 8850 meters tall
                elev_lim = (-1200.0, 9000.0)
            else:
                raise ValueError('limits=={0} not implemented'.format(limits))
        elif isinstance(limits, list) or isinstance(limits, tuple):
            if len(limits) != 2:
                raise ValueError('If given as a list or tuple, "limits" must have exactly 2 elements')
            try:
                elev_lim = (float(min(limits)), float(max(limits)))
            except ValueError:
                raise ValueError('All elements of limits must be convertible to floats')

        fig = plt.figure()
        plt.pcolormesh(elev, cmap='gray')
        plt.clim(elev_lim[0], elev_lim[1])
        plt.axes().set_aspect('equal','datalim')
        plt.axis('off')
        plt.savefig(filename, bbox_inches=0)
        plt.close(fig)