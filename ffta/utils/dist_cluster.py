# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 21:19:26 2018

@author: Raj
"""

import numpy as np
import sklearn as sk
import pycroscopy as px
from pycroscopy.processing.cluster import Cluster
from pycroscopy.processing.process import Process
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from ffta.utils import mask_utils, hdf_utils
from ffta import pixel
"""
Creates a Class with various data grouped based on distance to masked edges
A more general version of CPD clustering data

"""

class dist_cluster(object):

    def __init__(self, h5_main, parms_dict, data_avg = '', mask=None):
        """

        h5_file : h5Py dataset or str
            File type to be processed.

        mask : ndarray
            mask file to be loaded, is expected to be of 1 (transparent) and 0 (opaque)
            loaded e.g. mask = mask_utils.loadmask('E:/ORNL/20191221_BAPI/nice_masks/Mask_BAPI20_0008.txt')

        imgsize : list
            dimensions of the image

        ds_group : str or H5Py Group
            Where the target dataset is located

        light_on : list
            the time in milliseconds for when light is on

        Example usage:
        >> mymask = mask_utils.load_mask_txt('Path_mask.txt')
        >> CPD_clust = distance_utils.CPD_sluter(h5_path, mask=mymask)
        >> CPD_clust.analyze_CPD(CPD_clust.CPD_on_avg)

        """
#        hdf = px.ioHDF5(h5_file)
        self.h5_main = h5_main
        self.data_avg = px.hdf_utils.getDataSet(h5_main.parent, data_avg)[0].value

        # Set up datasets data
        self.data = self.h5_main[()]
        self.parms_dict = parms_dict
#        self.pnts_per_CPDpix = self.CPD.shape[1]

        self._params()

        # Create mask for grain boundaries

        #
        if not mask.any():
            mask = np.ones([self.num_rows, self.num_cols])

        self.mask = mask
        self.mask_nan, self.mask_on_1D, self.mask_off_1D = mask_utils.load_masks(self.mask)
        self._idx_1D = np.copy(self.mask_off_1D)

        return

    def _params(self):
        """ creates CPD averaged data and extracts parameters """

        parms_dict = self.parms_dict
        self.num_rows = parms_dict['num_rows']
        self.num_cols = parms_dict['num_cols']
        self.sampling_rate = parms_dict['sampling_rate']
        self.FastScanSize = parms_dict['FastScanSize']
        self.SlowScanSize = parms_dict['SlowScanSize']

#        IO_rate = parms_dict['IO_rate_[Hz]']     #sampling_rate
        self.pxl_time = parms_dict['total_time']    #seconds per pixel
        self.dt = self.pxl_time/self.data.shape[1]

        return

    def analyze(self):
        """
        Creates 1D arrays of data and masks
        Then, calculates the distances and saves those.

        This also creates CPD_scatter within the distances function

        Example usage:
        >> mymask = mask_utils.load_mask_txt('Path_mask.txt')
        >> CPD_clust = distance_utils.CPD_sluter(h5_path, mask=mymask)
        >> CPD_clust.analyze_CPD(CPD_clust.CPD_on_avg)

        """
        # Create 1D arrays
        self._data_1D_values(self.mask)
        self._make_distance_arrays()

        self.data_dist, _ = self._distances(self.data_1D_pos, self.mask_on_1D_pos)

        return

    def _data_1D_values(self, mask):
        """
        Uses 1D Mask file (with NaN and 0) and generates CPD of non-grain boundary regions

        h5_file : H5Py File
            commonly as hdf.file

        CPD : ndarray
            RowsXColumns matrix of CPD average values such as CPD_on_avg

        mask : ndarray, 2D
            Unmasked locations (indices) as 1D location

        CPD_1D_vals : CPD as a 1D array with pnts_per_CPD points (num_rows*num_cols X pnts_per_data)
        CPD_avg_1D_vals : Average CPD (CPD_on_avg, say) that is 1D

        """

        ones = np.where(mask == 1)
        
        self.data_avg_1D_vals = np.zeros(ones[0].shape[0])
        self.data_1D_vals = np.zeros([ones[0].shape[0], self.data.shape[1]])

        for r,c,x in zip(ones[0], ones[1], np.arange(self.data_avg_1D_vals.shape[0])):
            self.data_avg_1D_vals[x] = self.data_avg[r][c]
            self.data_1D_vals[x,:] = self.data[self.num_cols*r + c,:]

        return

    def _make_distance_arrays(self):
        """
        Generates 1D arrays where the coordinates are scaled to image dimenions

        Generates
        -------
        mask_on_1D_pos : ndarray Nx2
            Where mask is applied (e.g. grain boundaries)

        mask_off_1D_pos : ndarray Nx2
            Where mask isn't applied (e.g. grains)

        CPD_1D_pos : ndarray Nx2
            Identical to mask_off_1D_scaled, this exists just for easier bookkeeping
            without fear of overwriting one or the other

        """

        csz = self.FastScanSize / self.num_cols
        rsz = self.SlowScanSize / self.num_rows

        mask_on_1D_pos = np.zeros([self.mask_on_1D.shape[0],2])
        mask_off_1D_pos = np.zeros([self.mask_off_1D.shape[0],2])

        for x,y in zip(self.mask_on_1D, np.arange(mask_on_1D_pos.shape[0])):
            mask_on_1D_pos[y,0] = x[0] * rsz
            mask_on_1D_pos[y,1] = x[1] * csz

        for x,y in zip(self.mask_off_1D, np.arange(mask_off_1D_pos.shape[0])):
            mask_off_1D_pos[y,0] = x[0] * rsz
            mask_off_1D_pos[y,1] = x[1] * csz

        self.data_1D_pos = np.copy(mask_off_1D_pos) # to keep straight, but these are the same
        self.mask_on_1D_pos = mask_on_1D_pos
        self.mask_off_1D_pos = mask_off_1D_pos

        return


    def _distances(self, data_1D_pos, mask_on_1D_pos):
        """
        Calculates pairwise distance between CPD array and the mask on array.
        For each pixel, this generates a minimum distance that defines the "closeness" to
        a grain boundary in the mask

        Returns to Class
        ----------------
        CPD_scatter : distances x CPD_points (full CPD at each distance)
        CPD_avg_scatter : distances x 1 (CPD_average at each distance)

        """
        data_dist = np.zeros(data_1D_pos.shape[0])
        data_avg_dist = np.zeros(data_1D_pos.shape[0])

        # finds distance to nearest mask pixel
        for i, x in zip(data_1D_pos, np.arange(data_dist.shape[0])):

            d = sk.metrics.pairwise_distances([i], mask_on_1D_pos)
            data_dist[x] = np.min(d)
            data_avg_dist[x] = np.mean(d)

        # create single [x,y] dataset
        self.data_avg_scatter = np.zeros([data_dist.shape[0],2])
        for x,y,z in zip(data_dist, self.data_avg_1D_vals, np.arange(data_dist.shape[0])):
            self.data_avg_scatter[z] = [x, y]

        self.data_scatter = np.copy(self.data_1D_vals)
        self.data_scatter = np.insert(self.data_scatter, 0, data_dist, axis=1)

        return data_dist, data_avg_dist

    def kmeans(self, clusters=3, show_results=False, plot_mid=[]):

        """"
        Simple k-means

        Data typically is self.CPD_scatter

        plot_pts : list
            Index of where to plot (i.e. when light is on). Defaults to p_on:p_off

        Returns
        -------
        self.results : KMeans type

        self.segments : dict, Nclusters
            Contains the segmented arrays for displaying

        """

        data = self.data_scatter
        data_avg = self.data_avg_scatter

        # create single [x,y] dataset
        estimators = sk.cluster.KMeans(clusters)
        self.results = estimators.fit(data)

        if show_results:
            
            ax, fig = self.plot_kmeans(plot_mid=plot_mid)
            
            return self.results, ax, fig
        
        return self.results
    
    def plot_kmeans(self, plot_mid=[]):

        labels = self.results.labels_
        cluster_centers = self.results.cluster_centers_
        labels_unique = np.unique(labels)
        self.segments = {}
        self.clust_tfp = []

        if not any(plot_mid):
            plot_mid = [0, int(self.data_scatter.shape[1]/2)]

        # color defaults are blue, orange, green, red, purple...
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
                  '#d62728', '#9467bd', '#8c564b',
                  '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']

        fig, ax = plt.subplots(nrows=1, figsize=(12, 6))
        ax.set_xlabel('Distance to Nearest Boundary (um)')
        ax.set_ylabel('tfp (us)')

        for i in labels_unique:

            # CPD time trace
            ax.plot(self.data_scatter[labels==labels_unique[i],0]*1e6,
                    self.data_avg_scatter[labels==labels_unique[i],1]*1e6,
                     c=colors[i], linestyle='None', marker='.')

            pix = pixel.Pixel(cluster_centers[i],self.parms_dict)
            clust_tfp, _, _ = pix.analyze()
            self.clust_tfp.append(clust_tfp)

            ax.plot(cluster_centers[i][0]*1e6, 
                    np.mean(self.data_avg_scatter[labels==labels_unique[i],1]*1e6),
                     marker='o',markerfacecolor = colors[i], markersize=8,
                     markeredgecolor='k')

        return ax, fig


    def elbow_plot(self, data):
        """"
        Simple k-means elbow plot

        Data typically is self.data_scatter


        Returns
        -------
        self.results : KMeans type

        """

        Nc = range(1, 20)
        km = [sk.cluster.KMeans(n_clusters=i) for i in Nc]

        score = [km[i].fit(data).score(data) for i in range(len(km))]

        fig, ax = plt.subplots(nrows=1, figsize=(6, 4))
        ax.plot(Nc, score, 's', markersize=8)
        ax.set_xlabel('Clusters')
        ax.set_ylabel('Score')
        fig.tight_layout()

        return score

    def segment_maps(self, results=None):

        """
        This creates 2D maps of the segments to overlay on an image

        Returns to Class
        ----------------
        segments is in actual length
        segments_idx is in index coordinates
        segments_CPD is the full CPD trace (i.e. vs time)
        segments_CPD_avg is for the average CPD value trace (not vs time)

        To display, make sure to do [:,1], [:,0] given row, column ordering
        Also, segments_idx is to display since pyplot uses the index on the axis

        """
        
        if not results:
            results = self.results
        labels = results.labels_
        cluster_centers = results.cluster_centers_
        labels_unique = np.unique(labels)

        self.segments = {}
        self.segments_idx = {}
        self.segments_data = {}
        self.segments_data_avg = {}

        for i in range(len(labels_unique)):
            self.segments[i] = self.data_1D_pos[labels==labels_unique[i],:]
            self.segments_idx[i] = self._idx_1D[labels==labels_unique[i],:]
            self.segments_data[i] = self.data_1D_vals[labels==labels_unique[i],:]
            self.segments_data_avg[i] = self.data_avg_1D_vals[labels==labels_unique[i]]

        # the average CPD in that segment
        self.data_time_avg = {}
        for i in range(len(labels_unique)):

            self.data_time_avg[i] = np.mean(self.segments_data[i], axis=0)

        return

    def plot_segment_maps(self, ax, newImage=False):
        """ Plots the segments using a color map on given axis ax"""


        colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
                  '#d62728', '#9467bd', '#8c564b',
                  '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']

        if newImage:
            fig, ax = plt.subplots(nrows=1, figsize=(8, 6))
            im0,_ = px.plot_utils.plot_map(ax, self.data_on_avg, x_size=self.FastScanSize,
                                   y_size=self.SlowScanSize, show_cbar=False, 
                                   cmap='inferno')
           
        for i in self.segments_idx:

            im1, = ax.plot(self.segments_idx[i][:,1], self.segments_idx[i][:,0],
                          color=colors[i], marker='s', linestyle='None', label=i)

        ax.legend(fontsize=14, loc=[-0.18,0.3])

        if newImage:
            return im0, im1

        return im1

    def heat_map(self, bins=50):

        """
        Plots a heat map using CPD_avg_scatter data
        """

        heatmap, _, _ = np.histogram2d(self.CPD_avg_scatter[:,1],self.data_avg_scatter[:,0],bins)

        fig, ax = plt.subplots(nrows=1, figsize=(8, 6))
        ax.set_xlabel('Distance to Nearest Boundary (um)')
        ax.set_ylabel('CPD (V)')
        xr = [np.min(self.data_avg_scatter[:,0])*1e6, np.max(self.data_avg_scatter[:,0])*1e6]
        yr = [np.min(self.data_avg_scatter[:,1]), np.max(self.data_avg_scatter[:,1])]
        aspect = int((xr[1]-xr[0])/ (yr[1]-yr[0]))
        ax.imshow(heatmap, origin='lower', extent=[xr[0], xr[1], yr[0],yr[1]],
                   cmap='viridis', aspect=aspect)
        fig.tight_layout()

        return ax, fig


    def animated_clusters(self, clusters=3, one_color=False):

        plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\Raj\Downloads\ffmpeg-20180124-1948b76-win64-static\bin\ffmpeg.exe'

        fig, a = plt.subplots(nrows=1, figsize=(13,6))

        time = np.arange(0, self.pxl_time, 2*self.dtCPD)
        idx = np.arange(1, self.CPD.shape[1], 2) # in 2-slice increments

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
                  '#d62728', '#9467bd', '#8c564b',
                  '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']

        if one_color:
            colors = ['#1f77b4','#1f77b4','#1f77b4',
                      '#1f77b4','#1f77b4','#1f77b4',
                      '#1f77b4','#1f77b4','#1f77b4']

        a.set_xlabel('Distance to Nearest Boundary (um)')
        a.set_ylabel('CPD (V)')

        labels = self.results.labels_
        cluster_centers = self.results.cluster_centers_
        labels_unique = np.unique(labels)

        ims = []
        for t in idx:

            data = np.zeros([self.CPD_scatter.shape[0], 3])
            data[:,0] = self.CPD_scatter[:,0]
            data[:,1:3] = self.CPD_scatter[:, t:t+2]

            _results  = self.kmeans(data, clusters = clusters)

            labels = _results.labels_
            cluster_centers = _results.cluster_centers_
            labels_unique = np.unique(labels)

            km_ims = []

            for i in range(len(labels_unique)):

                tl0, = a.plot(data[labels==labels_unique[i],0]*1e6,data[labels==labels_unique[i],1],
                       c=colors[i],linestyle='None', marker='.')

                tl1, = a.plot(cluster_centers[i][0]*1e6, cluster_centers[i][1],
                         marker='o',markerfacecolor = colors[i], markersize=8,
                         markeredgecolor='k')

                ims.append([tl0, tl1])

            km_ims = [i for j in km_ims for i in j] # flattens
            ims.append(km_ims)

        ani = animation.ArtistAnimation(fig, ims, interval=120,repeat_delay=10)

        ani.save('kmeans_graph_.mp4')


        return


    def animated_image_clusters(self, clusters=5):
        """
        Takes an image and animates the clusters over time on the overlay

        As of 4/3/2018 this code is just a placeholder

        """

        plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\Raj\Downloads\ffmpeg-20180124-1948b76-win64-static\bin\ffmpeg.exe'

        fig, ax = plt.subplots(nrows=1, figsize=(13,6))
        im0 = px.plot_utils.plot_map(ax, self.CPD_on_avg, x_size=self.FastScanSize,
                               y_size=self.SlowScanSize, show_cbar=False, 
                               cmap='inferno')

        time = np.arange(0, self.pxl_time, 2*self.dtCPD)
        idx = np.arange(1, self.CPD.shape[1], 2) # in 2-slice increments

        ims = []
        for t in idx:

            data = np.zeros([self.CPD_scatter.shape[0], 3])
            data[:,0] = self.CPD_scatter[:,0]
            data[:,1:3] = self.CPD_scatter[:, t:t+2]

            _results  = self.kmeans(data, clusters = clusters)
            self.segment_maps(results=_results)
            im1 = self.plot_segment_maps(ax)            

            ims.append([im1])

        ani = animation.ArtistAnimation(fig, ims, interval=120,repeat_delay=10)

        ani.save('img_clusters_.mp4')


        return


