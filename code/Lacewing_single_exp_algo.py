import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

from scipy.signal import filtfilt, medfilt, convolve
from skimage.measure import label as label_connected_graph
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from pathlib import Path
from tqdm.auto import tqdm
from collections import OrderedDict
import struct

# for fitting
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

import time


def binary_file_read(file):
    """ Unpacks .bin file and returns list of the data in it

    Parameters
    ----------
    file : Path
        .bin file name
    Returns
    -------
    list
        data_list : list of the data in the input binary file
    """
    fw = open(file, 'rb')
    data_byte = fw.read()
    data_len = len(data_byte) >> 1
    data_list = []
    for n in range(0, data_len):
        (data, ) = struct.unpack('>H', data_byte[2*n:2*(n+1)])
        data_list.append(data)
    return data_list


def load_data(exp_path, version):
    """ Unpacks the .bin file contained in the folder exp_path and returns list of the data

    Parameters
    ----------
    exp_path : Path
        path of the folder containing the .bin file
    version : str
        version of the chip, 'v2' or 'v4'

    Returns
    -------
    list
        data : list of the data from the binary file in the input folder
    """
    if version == 'v4':
        filename = [i for i in exp_path.glob("VF*.bin")]
    elif version == 'v2':
        filename = [i for i in exp_path.glob("*.bin")]
    else:
        raise NotImplemented(f'The code for version {version} of the chip has not been implemented in load_data')

    if len(filename) > 0:
        filename = filename[0]
    else:
        print(f'No bin file found in {exp_path}')
    data = binary_file_read(filename)
    return data


def binary_to_numpy(data, f_start, f_end, version):
    ''' Converts list of data from bin file to numpy representation

    Parameters
    ----------
    data : list
        list of input data to be converted into numpy array
    f_start : int
        first frame to be considered
    f_end : int
        last frame to be considered. To set it change code. Now it's len(data)//4372
    version : str
        version of the chip, 'v2' or 'v4'

    Returns
    -------
    tuple
        (time_vect, frame_3d, temp_vect) where
        - time_vect : ndarray
            1D array containing the sampling times
        - frame_3d : ndarray
            3D array containing time frames of the sensor array outputs. NxMxT, where NxM is the sensor array, T is time
        - temp_vect : ndarray
            1D array containing some temperature information
    '''
    # convert data into 3D object [x, y, time]
    #f_end = len(data)//4372  # remove line when we move to GUI
    if version =='v2':
        N = 4372
    elif version == 'v4':
        N = 4374
    else:
        raise NotImplemented(f'The code for version {version} of the chip has not been implemented in binary_to_numpy')

    f_end = len(data)//N  # remove line when we move to GUI
    n_time = f_end - f_start + 1
    frame_3d = np.zeros((78, 56, n_time))
    time_vect = np.zeros(n_time)
    temp_vect = np.zeros(n_time)
    for i, n in enumerate(range(f_start - 1, f_end)):
        pack = data[n * N:(n + 1) * N]
        frame_1d = np.array(pack[:4368])
        frame_2d = frame_1d.reshape(78, 56, order='F')
        frame_3d[:, :, i] = frame_2d
        time_vect[i] = pack[4368] / 10
        temp_vect[i] = pack[4370]
    return time_vect, frame_3d, temp_vect


def split_chem_and_temp(arr):
    """ Separates temperature and chemical pixels.

    Parameters
    ----------
    arr : ndarray
        3D array of chemical and temperature pixels (1 temp pixel for each 8 chem pixels)

    Returns
    -------
    tuple
        (arr_temp, arr_chem) where
        - arr_temp - 3D array of the temperature pixels
        - arr_chem - 3D array of the chemical pixels, where the temperature pixels are replaced by a chemical pixel whose value is the average of the ones surrounding it
    """

    arr_temp = arr[1::3, 1::3, :]  # obtain the temperature array by selecting one pixel every 3

    mask = np.ones((3, 3, 1)) / 8
    mask[1, 1, 0] = 0
    # mask = 1 1 1
    #        1 0 1
    #        1 1 1
    # 2D convolution of a signal with the mask above
    # results in an output signal where each value is the average of the surrounding ones in the input signal
    av_3d = convolve(arr, mask, mode='same')  # perform convolution

    arr_chem = arr.copy()  # copy the original to preserve the original chemical pixels
    arr_chem[1::3, 1::3, :] = av_3d[1::3, 1::3, :]  # substitute the temp pixels with the average found by convolution

    return arr_temp, arr_chem


def split_into_n_wells(frame_3d, n=2):
    """ Given an input 3D array with data from the entire chip,
    splits the top and bottom well of the sensor array and returns two 2D arrays.

    Parameters
    ----------
    frame_3d : ndarray
        Input 3D numpy array of data to be split in wells
    n : int
        Number of wells. Default is 2. Now, only works for n=2

    Returns
    -------
    tuple
        (top_well, bot_well) where
        - top_well - 2D array (T x N'M) with data for the top well
        - bot_well - 2D array (T x N''M) with data for the bottom well

    """
    assert n == 2, "Only implemented 2 wells so far!"  # if input n is different from 2, output the error message
    n_time = frame_3d.shape[2]  # shape[2] is the number of samples in time
    top_well = frame_3d[:39, :, :].reshape(-1, n_time, order='C').T  # obtain the top well
    # and reshape the 3D matrix N' x M x T to a 2D matrix of dimensions T x N'M
    bot_well = frame_3d[-39:, :, :].reshape(-1, n_time, order='C').T  # obtain the bottom well
    # and reshape the 3D matrix N'' x M x T to a 2D matrix of dimensions T x N''M (N' + N'' = N)

    return top_well, bot_well


def binary_to_wells(data, version):
    """ Converts list of data from .bin file to dictionary with time, temperature and chemical data for the top and bottom wells.

    Parameters
    ----------
    data : list
        list of data extracted from .bin file
    version : str
        version of the chip, 'v2' or 'v4'

    Returns
    -------
    dictionary
        exp_data_unprocessed : dict with data of top well and bottom well ['Top Well']/['Bot Well'].
        Each contains time, chemical and temperature data ['Time']/['Chemical Data']/['Temperature Data']
    """
    # Get data from the bin file. Split chem/temp and top/bottom wells
    f_start, f_end = 1, 976  # This is set by Lei somewhere
    time_vect, frame_3d, temp_vect = binary_to_numpy(data, f_start, f_end, version)  # build data arrays from the binary file

    n_rows, n_cols, n_time = frame_3d.shape  # find data dimensions: NxMxT = rows of sensors x columns x samples
    assert n_time != 0, f'Could not find data for experiment {exp_path.stem}. Check the bin files.'

    arr_temp, arr_chem = split_chem_and_temp(frame_3d)  # separate chemical and temperature pixels
    top_well, bot_well = split_into_n_wells(arr_chem, n=2)  # split top and bottom wells for chemical pixels
    top_well_temp, bot_well_temp = split_into_n_wells(arr_temp, n=2)  # split top and bottom wells for temp pixels

    exp_data_unprocessed = OrderedDict({
        'Top Well': {
            'Time': time_vect,
            'Chemical Data': top_well,
            'Temperature Data': top_well_temp,
        },

        'Bot Well': {
            'Time': time_vect,
            'Chemical Data': bot_well,
            'Temperature Data': bot_well_temp,
        }
    })
    return exp_data_unprocessed


def get_initial_data(exp_data_unprocessed, version):
    """ From a dictionary of the complete experiment data, returns data until t_start. This simulates real-time implementation

    Parameters
    ----------
    exp_data_unprocessed : dict
        dictionary with unprocessed data
    version : str
        version of the chip, 'v2' or 'v4'

    Returns
    -------
    dict
        exp_data_initial : dictionary with the data until t_start
    """
    time_vect = exp_data_unprocessed['Top Well']['Time']
    top_well = exp_data_unprocessed['Top Well']['Chemical Data']
    bot_well = exp_data_unprocessed['Bot Well']['Chemical Data']
    top_well_temp = exp_data_unprocessed['Top Well']['Temperature Data']
    bot_well_temp = exp_data_unprocessed['Bot Well']['Temperature Data']

    # fig, ax = plt.subplots(2, 1, figsize=(12, 3*2))
    # ax[0].plot(time_vect, top_well)
    # ax[0].set_title('Chemical data. V4', fontsize=20)
    # ax[0].set_xlabel('Time (s)', fontsize=18)
    # ax[0].set_ylabel('Voltage (mV)', fontsize=18)
    #
    # ax[1].plot(time_vect, top_well_temp)
    # ax[1].set_title('Temperature data. V4', fontsize=20)
    # ax[1].set_xlabel('Time (s)', fontsize=18)
    # ax[1].set_ylabel('Voltage (mV)', fontsize=18)
    # plt.tight_layout()
    # plt.savefig('exp35_initialdata.eps')
    # plt.show()

    if version == 'v2':
        tstart_idx = time_to_index([930], time_vect)[0]
    elif version =='v4':
        tstart_idx = time_to_index([400], time_vect)[0]
    else:
        raise NotImplemented(f'The code for version {version} of the chip has not been implemented in get_initial_data')

    time_vect = time_vect[:tstart_idx]
    top_well = top_well[:tstart_idx, :]
    bot_well = bot_well[:tstart_idx, :]
    top_well_temp = top_well_temp[:tstart_idx, :]
    bot_well_temp = bot_well_temp[:tstart_idx, :]

    exp_data_initial = OrderedDict({
        'Top Well': {
            'Time': time_vect,
            'Chemical Data': top_well,
            'Temperature Data': top_well_temp,
        },

        'Bot Well': {
            'Time': time_vect,
            'Chemical Data': bot_well,
            'Temperature Data': bot_well_temp,
        }
    })
    return exp_data_initial


def get_current_data(exp_data_unprocessed, settled_idx_top, settled_idx_bot, tcurrent):
    """ From a dictionary of the complete experiment data, returns data from t_settled to t_current. This simulates real-time implementation

    Parameters
    ----------
    exp_data_unprocessed : dict
        dictionary with unprocessed data
    settled_idx_top : int
        index of the settled time for the top well
    settled_idx_bot : int
        index of the settled time for the bottom well
    tcurrent : int
        current time. Data will be returned until this point

    Returns
    -------
    dict
        exp_data_initial : dictionary with the data from t_settling to t_current
    """
    time_vect = exp_data_unprocessed['Top Well']['Time']
    top_well = exp_data_unprocessed['Top Well']['Chemical Data']
    bot_well = exp_data_unprocessed['Bot Well']['Chemical Data']
    top_well_temp = exp_data_unprocessed['Top Well']['Temperature Data']
    bot_well_temp = exp_data_unprocessed['Bot Well']['Temperature Data']

    tcurrent_idx_top = time_to_index([tcurrent+time_vect[settled_idx_top]], time_vect)[0]
    tcurrent_idx_bot = time_to_index([tcurrent+time_vect[settled_idx_bot]], time_vect)[0]
    time_vect_top = time_vect[settled_idx_top:tcurrent_idx_top] - time_vect[settled_idx_top]
    time_vect_bot = time_vect[settled_idx_bot:tcurrent_idx_bot] - time_vect[settled_idx_bot]
    top_well = top_well[settled_idx_top:tcurrent_idx_top, :]
    bot_well = bot_well[settled_idx_bot:tcurrent_idx_bot, :]
    top_well_temp = top_well_temp[settled_idx_top:tcurrent_idx_top, :]
    bot_well_temp = bot_well_temp[settled_idx_bot:tcurrent_idx_bot, :]

    # fig, ax = plt.subplots(figsize=(12, 3))
    # ax.plot(time_vect_top, top_well)
    # ax.set_title('Original data after settling. V4 Experiment 35', fontsize=20)
    # ax.set_xlabel('Time (s)', fontsize=18)
    # ax.set_ylabel('Voltage (mV)', fontsize=18)
    # # plt.savefig('exp35_currentdata.eps')
    # plt.show()

    exp_data_current = OrderedDict({
        'Top Well': {
            'Time': time_vect_top,
            'Chemical Data': top_well,
            'Temperature Data': top_well_temp,
        },

        'Bot Well': {
            'Time': time_vect_bot,
            'Chemical Data': bot_well,
            'Temperature Data': bot_well_temp,
        }
    })
    return exp_data_current


def filter_by_vref(X, v_thresh=70):
    '''
    Identifies active pixels by checking if one of the first 10 derivatives d(i) is > v_thresh

    Parameters
    ---------
    X : ndarray
        Input 2D array (T x NM). T = time samples, NM = total number of pixels
    v_thresh : int, optional
        Minimum value of the derivative d(i)=X(i+1)-X(i) in mV. Default is 70

    Returns
    -------
    ndarray
        1D array of bool with dimension (NM). For each pixel, returns True if, during the first 10 samples,
        one of the derivatives is > v_thresh. The derivatives are calculated as d(i) = X(i+1)-X(i)
    '''
    return (np.diff(X[:10, :], axis=0) > v_thresh).any(axis=0)  # check if one of the first 10 derivatives is >v_thresh


def filter_by_vrange(X, v_range=(100, 900)):
    '''
    Identifies active pixels by checking that all the values are in v_range

    Parameters
    ---------
    X : ndarray
        Input 2D array (T x NM). T = time samples, NM = total number of pixels
    v_range : (int, int), optional
        tuple containing the minimum and maximum allowable voltage in mV. Default is (100, 900)

    Returns
    -------
    ndarray
        1D array of bool with dimension (NM). For each pixel, returns True if the value is always in v_range
    '''
    return (X < v_range[1]).all(axis=0) & (X > v_range[0]).all(axis=0)  # for each pixel, check if all the values are
    # within the given range


def filter_by_derivative(X, vthresh=5):
    """ Identifies active pixels by checking that the absolute value of the derivative is always below vthresh

    Parameters
    ----------
    X : ndarray
        input 2D array of shape TxNM
    vthresh : int
        threshold for active pixels. Default is 5

    Returns
    -------
    ndarray
        1D array of bool with dimension (NM). For each pixel, returns True if all the derivatives are below vthresh
    """
    x_diff = np.abs(np.diff(X, axis=0))
    return (x_diff < vthresh).all(axis=0)


def time_to_index(times, time_vect):
    '''
    Returns index of the times closest to the desired ones time_vect

    Arguments
    ---------
    times : list
        list of integers containing the desired times
    time_vect : nparray
        array of the times at which the values are sampled

    Returns
    -------
    list
        for each element in the input list times, return an element in the output list
        with the index of the sample closest to the desired time
    '''
    indices = []
    for time in times:  # for each time in the input list
        indices.append( np.argmin(np.abs(time_vect - time)) )
        # find index of the sampled time (in time_vect) closest to the desired one (time)
    return indices


def find_loading_time(time_vect, X, bounds=(600, 900), viz=False):  # for v2
    ''' Finds loading and settling time for the data of v2 chip

    Parameters
    ----------
    time_vect : ndarray
        1D array with dimension T containing the sampling times
    X : ndarray
        2D array with dimension TxNM containing the sampled data
    bounds : list, optional
        tuple containing the minimum and maximum times (in ms) where the loading time has to be searched.
        Default is (600, 900)
    viz : bool, optional
        if viz=True, show the plot. Default is False

    Returns
    -------
    tuple
        - settled_index : index at which the settling occurs
        - settled_time : time at which the settling occurs
    '''

    search_start, search_end = time_to_index(bounds, time_vect)  # for each time in bounds, find the index
    # of the sample (in time_vect) that is closest to the desired one (in bounds)
    X_mean = np.mean(X, axis=1)  # for each sample, calculate the mean of all pixels
    X_mean_diff = np.diff(X_mean)  # find the derivative

    loading_index = np.argmax(X_mean_diff[search_start:search_end]) + search_start + 1  # find the index
    # where the derivative is max in the specified interval
    loading_index = loading_index  # add settling time
    settled_index = loading_index + 10  # add settling time
    settled_time = time_vect[settled_index]  # find the time that index corresponds to

    if viz:  # if viz is true, plot the following
        fig, ax = plt.subplots(3, 1)
        fig.suptitle('Finding Loading Time...')

        ax[0].set(title='Active Chemical Pixels, ACP')
        ax[0].plot(time_vect, X)  # plot the active chemical pixels

        ax[1].set(title='Mean(ACP)')
        ax[1].plot(time_vect, X_mean)  # plot the average of the pixels
        ax[1].axvline(time_vect[search_start], color='C1')  # plot vertical line: beginning of the interval
        ax[1].axvline(time_vect[search_end], color='C1')  # plot vertical line: end of the interval
        ax[1].axvline(settled_time, color='C2')  # plot vertical line: the loading time that was found

        ax[2].set(title='Diff(Mean(ACP))')
        ax[2].plot(time_vect[1:], X_mean_diff)  # plot the derivative of the mean
        ax[2].axvline(time_vect[search_start], color='C1')  # plot vertical line: beginning of the interval
        ax[2].axvline(time_vect[search_end], color='C1')  # plot vertical line: end of the interval
        ax[2].axvline(settled_time, color='C2')  # plot vertical line: the loading time that was found

        plt.tight_layout()
        plt.show()
    return settled_index, settled_time


def find_settled_time(time_vect, temp):  # for v4
    """ Finds the settling time for the data of v4 of the chip

    Parameters
    ----------
    time_vect : ndarray
        time data
    temp : ndarray
        temperature data
    Returns
    -------
    tuple
        - settled_index : index at which the settling occurs
        - settled_time : time at which the settling occurs
    """
    temp_mean = np.mean(temp, axis=1)
    threshold = temp_mean[0]+0.95*(temp_mean[-1]-temp_mean[0])
    settled_idx = np.argmax(temp_mean > threshold)
    return settled_idx, time_vect[settled_idx]  # return the index and the time that corresponds to


def cleanup_pixels(idx_active):
    """Given an input 1D array representing which pixels in a 2D grid are active,
    return a 1D array representing which pixels belong to the largest 2D region of active pixels

    Parameters
    ----------
    idx_active : ndarray
        1D array of bool with dimension NM (number of pixels), where True means that the pixel is active

    Returns
    -------
    1D array of bool with dimension NM. For each pixel, True if is in the largest active region
    """
    idx_2d = idx_active.reshape(-1, 56)
    labels_2d = label_connected_graph(idx_2d, background=0)
    # Pixels are connected when they are neighbors and have the same value (active or inactive)
    # Each active region is given a 'name' (int). Pixels in active regions have non-zero names.
    # Inactive pixels are considered background and therefore are all '0'

    values, counts = np.unique(labels_2d.reshape(-1), return_counts=True)  # return the region names (values)
    # and how many pixels are in each region (counts)
    ind_back = values == 0  # ind_back is a list of bool where the only True value corresponds to the inactive region
    values, counts = values[~ind_back], counts[~ind_back]  # tilde = bitwise negation.
    # return non-zero region names and how many pixels each. (recall Zero-regions are inactive)

    ind = np.argmax(counts)  # Find the number of pixels in the largest active region
    max_idx = values[ind]  # Find the 'name' of the largest active region

    return (labels_2d == max_idx).reshape(-1)  # Identify the largest active region.
    # Only the pixels in the largest active region are 'True'


def largest_cluster(model):
    '''
    Identifies the largest cluster

    Parameters
    ----------
    model :
        model of cluster

    Returns
    -------
    model.label_
        return label of the largest cluster, i.e. the cluster with the largest number of pixels
    '''
    values, counts = np.unique(model.labels_, return_counts=True)  # identify labels and number of pixels in each
    return values[np.argmax(counts)]  # return the label of the largest region


def count_pixels_in_polygon(x_coord, y_coord, polygon):
    """ Given a set of points defined by (x_coord, y_coord), counts the number of input points that are in the polygon

    Parameters
    ----------
    x_coord : x coordinates of the points
    y_coord : y coordinates of the points
    polygon : input polygon

    Returns
    -------
    number of input pixels that are in the polygon

    """
    count = 0
    for x, y in zip(x_coord, y_coord):  # for every point
        point = Point(x, y)
        count += polygon.contains(point)  # add one if the pixel of the cluster is in the polygon
    return count  # return number of pixels (x_coord, y_coord) in the polygon


def center_cluster(model, idx_active, viz=False):
    '''
    Identifies the center cluster, i.e. the cluster with more active pixels in the centre region where the well is

    Parameters
    ----------
    model :
        model of cluster
    idx_active : ndarray
        1D array with dimension TxNM indicating the active pixels
    viz : bool, optional
        if viz=True, show thw plot. Default is False

    Returns
    -------
    model.label_
        return label of the cluster that has more active pixels in the center
    '''
    y1, y2, = 20, 56
    x1, x2 = 9, 29
    polygon = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])

    best_count = -1
    for label in np.unique(model.labels_):  # for every cluster, check the one that has more active pixels in the poly
        temp = idx_active.copy()
        temp[temp] = model.labels_ == label
        x_coord, y_coord = np.where(temp.reshape(-1, 56))
        if viz:
            plt.plot(*polygon.exterior.xy)
            plt.plot(x_coord, y_coord, "x")
            plt.xlim((0, 39))
            plt.ylim((0, 56))
            plt.show()

        count = count_pixels_in_polygon(x_coord, y_coord, polygon)  # for each cluster, count the active pixels in poly
        if count > best_count:
            best_count = count
            best_cluster = label

    return best_cluster  # return the label of the cluster with more active pixels in the poly


def cleanup_by_kmeans(well, idx_active, n_max=6, method="center"):
    '''
    Identifies active pixels by k-means clustering

    Parameters
    ----------
    well : ndarray
        2D array of int with dimension T x NM containing the data for each pixel
    idx_active : ndarray
        1D array of bool with dimension NM indicating the active pixels
    n_max : int, optional
        Maximum number of clusters. The function finds the best number of clusters between 2 and n_max. Default is 6
    method : string, optional
        "largest" or "center". Default is "center"

    Returns
    -------
    ndarray
        1D array of bool indicating the active pixels
    '''

    #  k-means clustering is a method of vector quantization used to
    #  partition n observations into k clusters.
    #  Each observation belongs to the cluster with the nearest mean (cluster centers).

    X = well[:, idx_active].T  # X.shape = (n of active pixels)xT
    # X = X / np.linalg.norm(X, axis=1).reshape(-1, 1)  # TODO commented this. not ok

    best_score = -1
    for n_clusters in range(2, n_max+1):  # find the optimal number of clusters
        model = KMeans(n_clusters=n_clusters)  # build model for clustering
        model.fit(X)  # fit linear model
        score = silhouette_score(X, model.labels_)  # score is the mean Silhouette Coefficient over all samples
        # -1 <= score <= 1. 1 is the best, 0 indicates overlapping clusters, -1 indicates wrong assignment to cluster
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters  # at the end of the loop, this is optimal number of clusters

    print(f"The number of clusters found: {best_n_clusters}")
    model = KMeans(n_clusters=best_n_clusters)  # build the model
    model.fit(X)  # fit linear model
    if method == "largest":
        idx_cluster = largest_cluster(model)  # identify the largest cluster
    elif method == "center":
        idx_cluster = center_cluster(model, idx_active, viz=False)
    else:
        raise NotImplemented(f"The method {method} has not been implemented!")  # raise message if the desired method
        # is not implemented

    idx_active[idx_active] = idx_cluster == model.labels_  # active pixels are those in the largest/center cluster

    return idx_active


def decaying_exp(x, a, b):
    """ Returns exponential function

    Parameters
    ----------
    x : ndarray
        times
    a : double
        t(inf) value
    b : double
        slope to t=0

    Returns
    -------
    ndarray
        y-axis values of the function
    """
    return a*(1-np.exp(-b * x))


def fit_pixels_interpolate(time, X, idx_active, interpolate_idx):
    """ Interpolates the curves for each pixel

    Parameters
    ----------
    time : ndarray
        times
    X : ndarray
        TxNM array to be interpolated
    idx_active : ndarray
        NM array specifying pixels that are active
    interpolate_idx : int
        interpolation is performed until this index

    Returns
    -------
    popt : ndarray
        optimal parameters for interpolation of each pixel, with shape 2xNM
    """
    popt = np.zeros((2, X.shape[1]))

    # for every pixel
    for i in range(X.shape[1]):
        if idx_active[i]:  # if the pixel is active...
            data = filtfilt(b=np.ones(10) / 10, a=[1], x=X[:, i])

            # Fit the curve (interpolate)
            try:
                popt[:, i], pcov = curve_fit(decaying_exp, time[:interpolate_idx], data[:interpolate_idx], p0=[-10, 0.1])
            except:
                # print('EXCEPT: could not fit this pixel')
                popt[:, i] = 100
        else:
            popt[:, i] = 100
    return popt


def fit_pixels_extrapolate(time, X, idx_active, extrapolate_idx, drift_idx, popt, thresh_active=12):
    """ Interpolates the curves for each pixel, and find active pixels as those with fitting error below thresh_active

    Parameters
    ----------
    time : ndarray
        times
    X : ndarray
        TxNM array to be interpolated
    idx_active : ndarray
        NM array specifying pixels that are active
    extrapolate_idx : int
        extrapolation is performed until this index
    drift_idx : int
        drift magnitude is calculated until this index
    popt : ndarray
        optimal parameters for each pixel, shape 2xNM
    thresh_active : double, optional
        pixels are considered active if the fititng error is below thresh_active

    Returns
    -------
    tuple
        (idx_active, fit_error) where
        - idx_active : is NM array, where True indicates that the pixel is active
        - fit_error : fitting error for each pixel
    """
    idx_active_fit = np.array(np.zeros(X.shape[1]), dtype=bool)
    fit_error = np.array(np.zeros((extrapolate_idx, X.shape[1])))

    # for every pixel
    for i in range(X.shape[1]):
        if idx_active[i]:  # if the pixel is active...
            data = filtfilt(b=np.ones(10) / 10, a=[1], x=X[:, i])
            drift = data[drift_idx] - data[0]
            drift_magnitude = np.abs(drift)  # Find drift magnitude (in V)

            # Fit the curve (extrapolate)
            if drift_magnitude != 0 and popt[1, i] < 100:
                data_fit = decaying_exp(time[:extrapolate_idx], *popt[:,i])
                fit_error[:, i] = np.abs(data[:extrapolate_idx] - data_fit)
                idx_active_fit[i] = (fit_error[:, i] < thresh_active).all()
            else:
                idx_active_fit[i] = False
        else:
            idx_active_fit[i] = False
    idx_active = idx_active & idx_active_fit
    fit_error[:, ~idx_active] = 0
    return idx_active, fit_error


def preprocessing_initial(exp_path, exp_data_initial, version, visualise_steps=False):
    time_vect = exp_data_initial['Top Well']['Time']
    top_well = exp_data_initial['Top Well']['Chemical Data']
    bot_well = exp_data_initial['Bot Well']['Chemical Data']
    top_well_temp = exp_data_initial['Top Well']['Temperature Data']
    bot_well_temp = exp_data_initial['Bot Well']['Temperature Data']

    if visualise_steps:
        idx_top_active_vref = filter_by_vref(top_well)
        idx_bot_active_vref = filter_by_vref(bot_well)

    find_active_pixels = lambda x: filter_by_vref(x) & filter_by_vrange(x)
    # filter_by_vref finds pixels that are in contact with the solution
    # filter_by_vrange finds the pixels that have values always within a certain range,
    #   i.e. their value does not decay nor saturate
    # here, define find_active_pixels to find the pixels that satisfy both of the above conditions
    idx_top_active, idx_bot_active = [find_active_pixels(i) for i in [top_well, bot_well]]
    # find active pixels for both the top and bottom well

    if visualise_steps:
        fig, ax = plt.subplots(3, 2, figsize=(10, 3*3))
        exp_path_str = str(exp_path)
        experiment_id = exp_path_str[exp_path_str.rfind('_') + 1:]
        fig.suptitle(f'Preprocessing Initial. Experiment {experiment_id}', fontsize=22)
        ax[0, 0].imshow(idx_top_active_vref.reshape(-1, 56), cmap='cividis')
        ax[0, 0].set_title('Step 1: Filtered by Vref', fontsize=20)

        ax[0, 1].plot(time_vect, top_well[:, idx_top_active_vref])
        ax[0, 1].set_xlabel('Time (s)', fontsize=18)
        ax[0, 1].set_ylabel('Voltage (mV)', fontsize=18)

        ax[1, 0].imshow(idx_top_active.reshape(-1, 56), cmap='cividis')
        ax[1, 0].set_title('Step 2: Filtered by Vrange', fontsize=20)

        ax[1, 1].plot(time_vect, top_well[:, idx_top_active])
        ax[1, 1].set_xlabel('Time (s)', fontsize=18)
        ax[1, 1].set_ylabel('Voltage (mV)', fontsize=18)

    # for both wells, find largest active region
    idx_top_active, idx_bot_active = cleanup_pixels(idx_top_active), cleanup_pixels(idx_bot_active)

    if visualise_steps:
        ax[2, 0].imshow(idx_top_active.reshape(-1, 56), cmap='cividis')
        ax[2, 0].set_title('Step 3: Filtered spatially', fontsize=20)

        ax[2, 1].plot(time_vect, top_well[:, idx_top_active])
        ax[2, 1].set_xlabel('Time (s)', fontsize=18)
        ax[2, 1].set_ylabel('Voltage (mV)', fontsize=18)
        plt.tight_layout()
        # plt.savefig('exp35_preprocinitial.eps')
        plt.show()

    assert idx_top_active.sum() != 0, f'No active pixels in the top well. Experiment {exp_path.stem} invalid.'
    assert idx_bot_active.sum() != 0, f'No active pixels in the bot well. Experiment {exp_path.stem} invalid.'

    if version == 'v2':
        settled_idx_top, settled_t_top = find_loading_time(time_vect, top_well[:, idx_top_active], bounds=(600, 900))
        settled_idx_bot, settled_t_bot = find_loading_time(time_vect, bot_well[:, idx_bot_active], bounds=(600, 900))
    elif version == 'v4':
        settled_idx_top, settled_t_top = find_settled_time(time_vect, top_well_temp)
        settled_idx_bot, settled_t_bot = find_settled_time(time_vect, bot_well_temp)
    else:
        raise NotImplemented(f'The code for version {version} of the chip has not been implemented in preprocessing_initial')

    if visualise_steps:
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), dpi=300)
        ax[0].plot(time_vect, top_well_temp)
        ax[0].axvline(x=time_vect[settled_idx_top], color='k', linewidth=2, label='Settled time')
        ax[0].set_xlabel('Time (s)', fontsize=18)
        ax[0].set_ylabel('Voltage (mV)', fontsize=18)
        ax[0].set_title(f'Temperature data (until t_start). V4', fontsize=20)
        ax[0].legend(fontsize=18, bbox_to_anchor=(1, 1))

        ax[1].plot(time_vect, top_well)
        ax[1].axvline(x=time_vect[settled_idx_top-10], color='k', linestyle='--', linewidth=2, label='Loading time')
        ax[1].axvline(x=time_vect[settled_idx_top], color='k', linewidth=2, label='Settled time')
        ax[1].set_title(f'Chemical data (until t_start). V4', fontsize=20)
        ax[1].set_title(f'Original data time series (until t_start). Experiment {experiment_id}', fontsize=20)
        ax[1].set_xlabel('Time (s)', fontsize=18)
        ax[1].set_ylabel('Voltage (mV)', fontsize=18)
        ax[1].legend(fontsize=18, bbox_to_anchor=(1, 1))
        plt.tight_layout()
        # plt.savefig('exp35_data_preprocinitial.eps')
        plt.show()

    x_top = time_vect[settled_idx_top:] - time_vect[settled_idx_top]
    x_bot = time_vect[settled_idx_bot:] - time_vect[settled_idx_bot]
    Y_top = top_well[settled_idx_top:, :]
    Y_bot = bot_well[settled_idx_bot:, :]
    temp_top, temp_bot = top_well_temp[settled_idx_top:], bot_well_temp[settled_idx_bot:]

    # Save data for the experiment
    exp_data = OrderedDict({
        'Top Well': {
            'Active Pixels': idx_top_active,
            'Settled idx': settled_idx_top
        },

        'Bot Well': {
            'Active Pixels': idx_bot_active,
            'Settled idx': settled_idx_bot
        }
    })
    return exp_data


def preprocessing_update(exp_path, exp_data_current, exp_data_preproc, tinitial, visualise_plt=False, save_plt=False, visualise_steps=False):
    f_start, f_end = 1, 976  # This is set by Lei somewhere

    idx_top_active = exp_data_preproc['Top Well']['Active Pixels']
    idx_bot_active = exp_data_preproc['Bot Well']['Active Pixels']

    x_top = exp_data_current['Top Well']['Time']
    x_bot = exp_data_current['Bot Well']['Time']
    Y_top = exp_data_current['Top Well']['Chemical Data']
    Y_bot = exp_data_current['Bot Well']['Chemical Data']
    temp_top = exp_data_current['Top Well']['Temperature Data']
    temp_bot = exp_data_current['Bot Well']['Temperature Data']

    if visualise_steps:
        fig, ax = plt.subplots(5, 2, figsize=(10, 3*5), dpi=300)
        exp_path_str = str(exp_path)
        experiment_id = exp_path_str[exp_path_str.rfind('_') + 1:]
        fig.suptitle(f'Preprocessing Update. Experiment {experiment_id}', fontsize=22)
        ax[0, 0].imshow(idx_top_active.reshape(-1, 56), cmap='cividis')
        ax[0, 0].set_title('Step 0: From preprocessing_initial', fontsize=20)

        ax[0, 1].plot(x_top, Y_top[:, idx_top_active] - np.mean(Y_top[:3, idx_top_active], axis=0))
        ax[0, 1].set_xlabel('Time (s)', fontsize=18)
        ax[0, 1].set_ylabel('Voltage (mV)', fontsize=18)

    idx_top_active_range, idx_bot_active_range = [filter_by_vrange(i) for i in [Y_top, Y_bot]]
    idx_top_active = idx_top_active & idx_top_active_range
    idx_bot_active = idx_bot_active & idx_bot_active_range

    if visualise_steps:
        ax[1, 0].imshow(idx_top_active.reshape(-1, 56), cmap='cividis')
        ax[1, 0].set_title('Step 1: Filtered by Vrange', fontsize=20)

        ax[1, 1].plot(x_top, Y_top[:, idx_top_active] - np.mean(Y_top[:3, idx_top_active], axis=0))
        ax[1, 1].set_xlabel('Time (s)', fontsize=18)
        ax[1, 1].set_ylabel('Voltage (mV)', fontsize=18)

    assert idx_top_active.sum() != 0, f'No active pixels in the top well. Experiment {exp_path.stem} invalid.'
    assert idx_bot_active.sum() != 0, f'No active pixels in the bot well. Experiment {exp_path.stem} invalid.'

    # Filter pixels that are noisy (derivative above a threshold value)
    idx_top_active_der, idx_bot_active_der = filter_by_derivative(Y_top), filter_by_derivative(Y_bot)
    idx_top_active = idx_top_active & idx_top_active_der
    idx_bot_active = idx_bot_active & idx_bot_active_der

    if visualise_steps:
        ax[2, 0].imshow(idx_top_active.reshape(-1, 56), cmap='cividis')
        ax[2, 0].set_title('Step 2: Filtered by derivative', fontsize=20)

        ax[2, 1].plot(x_top, Y_top[:, idx_top_active] - np.mean(Y_top[:3, idx_top_active], axis=0))
        ax[2, 1].set_xlabel('Time (s)', fontsize=18)
        ax[2, 1].set_ylabel('Voltage (mV)', fontsize=18)

    # Find idx for 9-19 min bounds
    tinitial_idx_top = time_to_index([tinitial], x_top)[0]
    tinitial_idx_bot = time_to_index([tinitial], x_bot)[0]
    tcurrent_idx_top = Y_top.shape[0]
    tcurrent_idx_bot = Y_bot.shape[0]

    Y_top_bs = Y_top - np.mean(Y_top[:3, :], axis=0)
    Y_bot_bs = Y_bot - np.mean(Y_bot[:3, :], axis=0)

    # Filter pixels by k-means clustering
    if idx_top_active.sum() > 6:
        idx_top_active = cleanup_by_kmeans(Y_top_bs, idx_top_active)
    if idx_bot_active.sum() > 6:
        idx_bot_active = cleanup_by_kmeans(Y_bot_bs, idx_bot_active)

    if visualise_steps:
        ax[3, 0].imshow(idx_top_active.reshape(-1, 56), cmap='cividis')
        ax[3, 0].set_title('Step 3: Filtered by kmeans', fontsize=20)

        ax[3, 1].plot(x_top, Y_top_bs[:, idx_top_active])
        ax[3, 1].set_xlabel('Time (s)', fontsize=18)
        ax[3, 1].set_ylabel('Voltage (mV)', fontsize=18)

    # Further cleanup of pixels
    # idx_top_active, idx_bot_active = cleanup_pixels(idx_top_active), cleanup_pixels(idx_bot_active)

    # Fitting: interpolate signal
    popt_top = fit_pixels_interpolate(x_top, Y_top_bs, idx_top_active, tinitial_idx_top)
    idx_top_active, _ = fit_pixels_extrapolate(x_top, Y_top_bs, idx_top_active, tinitial_idx_top,
                                               tinitial_idx_top - 1, popt_top)
    popt_bot = fit_pixels_interpolate(x_bot, Y_bot_bs, idx_bot_active, tinitial_idx_bot)
    idx_bot_active, _ = fit_pixels_extrapolate(x_bot, Y_bot_bs, idx_bot_active, tinitial_idx_bot,
                                               tinitial_idx_bot - 1, popt_bot)

    if visualise_steps:
        ax[4, 0].imshow(idx_top_active.reshape(-1, 56), cmap='cividis')
        ax[4, 0].set_title('Step 4: Filtered by fitting', fontsize=20)

        ax[4, 1].plot(x_top, Y_top_bs[:, idx_top_active])
        ax[4, 1].set_xlabel('Time (s)', fontsize=18)
        ax[4, 1].set_ylabel('Voltage (mV)', fontsize=18)
        plt.tight_layout()
        # plt.savefig('exp35_preprocupdate.png')
        plt.show()

    if idx_top_active.sum() == 0:
        print(f'No active pixels in the top well. Experiment {exp_path.stem} invalid.')
    if idx_bot_active.sum() == 0:
        print(f'No active pixels in the bottom well. Experiment {exp_path.stem} invalid.')

    # Save data for the experiment
    exp_data_preproc = OrderedDict({
        'Top Well': {
            'Active Pixels': idx_top_active
        },

        'Bot Well': {
            'Active Pixels': idx_bot_active
        }
    })

    if visualise_plt:
        N_axes = 5  # plots for the top and bottom well
        fig, ax = plt.subplots(N_axes, 2, figsize=(10, 3 * N_axes))
        fig.suptitle(f"Preprocessing - Experiment {exp_path.stem}", fontsize=22)
        ax_top, ax_bot = ax[:, 0], ax[:, 1]

        for ax, x, Y, idx_active, temp, label in [(ax_top, x_top, Y_top[:, idx_top_active], idx_top_active, temp_top, 'Top'),
                                                  (ax_bot, x_bot, Y_bot[:, idx_bot_active], idx_bot_active, temp_bot, 'Bot')]:
            ax[0].set_title('Temp Pixels, TP', fontsize=20)
            ax[0].set_xlabel('Time (s)', fontsize=18)
            ax[0].set_ylabel('Voltage (mV)', fontsize=18)
            ax[0].plot(x, temp)  # plot temperature in time

            ax[1].set_title(f'{label} Well', fontsize=20)
            ax[1].imshow(idx_active.reshape(-1, 56), cmap='cividis')  # plot active/inactive pixels

            ax[2].set_title('Active Chemical Pixels, ACP', fontsize=20)
            if Y.shape[1] > 0:
                ax[2].plot(x, Y)  # plot chem pixels in time if there are active pixels
            ax[2].set_xlabel('Time (s)', fontsize=18)
            ax[2].set_ylabel('Voltage (mV)', fontsize=18)

            Y_bs = Y - np.mean(Y[:3, :], axis=0)  # remove the background by subtracting the first value
            ax[3].set_title('Background-subtracted ACP', fontsize=20)
            if Y_bs.shape[1] > 0:
                ax[3].plot(x, Y_bs)
                ax[3].plot(x, np.mean(Y_bs, axis=1), lw=2, color="k", label='Mean')  # plot the the signal with no offset
                ax[3].legend(fontsize=16)
            ax[3].set_xlabel('Time (s)', fontsize=18)
            ax[3].set_ylabel('Voltage (mV)', fontsize=18)

            Y_bs_smooth = pd.DataFrame(Y_bs).rolling(30).mean().values  # filter the data with a MA(30)
            # i.e. to smooth out the signal, each value is the average of the 30 surrounding ones
            Y_bs_smooth_mean = np.mean(Y_bs_smooth, axis=1)
            ax[4].set_title('MA(30) filtered ACP', fontsize=20)
            if Y_bs.shape[1] > 0:
                ax[4].plot(x, Y_bs_smooth)  # plot the smoothed signal
                ax[4].plot(x, Y_bs_smooth_mean, lw=2, color="k", label='Mean')
                ax[4].legend(fontsize=16)
            ax[4].set_xlabel('Time (s)', fontsize=18)
            ax[4].set_ylabel('Voltage (mV)', fontsize=18)

            ax[1].get_shared_x_axes().join(*ax[2:5])

        plt.tight_layout()
        if save_plt:
            exp_path_str = str(exp_path)
            experiment_id = exp_path_str[exp_path_str.rfind('_') + 1:]
            plt.savefig(Path(exp_path, f'preprocessing_{experiment_id}.eps'))
            print(f'Saved preprocessing figure for {exp_path.stem}')
        plt.show()

    return exp_data_preproc


def myfilt(arr, MED_FILTER=5, MA_FILTER_DERIV=40):
    ''' Filter input signal arr with median filter of order MED_FILTER (default 5)
    and moving average filter of order MA_FILTER_CERIV (default 40) '''
    arr_med = medfilt(arr, kernel_size=MED_FILTER)
    arr_med_ma = filtfilt(b=np.ones(MA_FILTER_DERIV) / MA_FILTER_DERIV, a=[1], x=arr_med)
    return arr_med_ma


def find_infl_points(array, time, tinitial, tfinal, D_RANGE = 20):
    ''' given the 2nd derivative as input , find inflection points of a signal
    and classify them in positive (indicating positive output) and negative '''
    # Find inflection points
    infls = np.where(np.diff(np.sign(array)))[0]
    # Only consider inflection points between tinitial and tfinal (that can indicate a positive sample)
    infls = [x for x in infls if tinitial <= time[x] <= tfinal]
    # Classify inflection points
    positive_infl_idx = []
    negative_infl_idx = []
    for k, infl in enumerate(infls, 1):
        if len(array) >= infl + D_RANGE and \
                all(d > 0 for d in array[infl - D_RANGE:infl - 1]) and \
                all(d < 0 for d in array[infl + 1:infl + D_RANGE]):
            positive_infl_idx.append(infl)
        else:
            negative_infl_idx.append(infl)
    return positive_infl_idx, negative_infl_idx


def processing(exp_path, exp_data_current, exp_data_preproc, tinitial, tfinal, visualise_plt=False, save_plt=False, visualise_steps=False):

    # Get string of experiment ID
    exp_path_str = exp_path.stem
    experiment_id = exp_path_str[exp_path_str.rfind('_') + 1:]
    n_wells = len(exp_data_preproc)

    well_summary = {}

    if visualise_plt:
        fig, axs = plt.subplots(5, n_wells, figsize=(10, 5 * 3), dpi=100)
        fig.suptitle(f'Processing - Experiment {exp_path_str}', fontsize=22)

    # For every well...
    for i, (well_name, well) in enumerate(exp_data_preproc.items()):
        # Get preprocessing data
        idx_active = well['Active Pixels']

        time = exp_data_current[well_name]['Time']
        chem_data = exp_data_current[well_name]['Chemical Data']
        temp_data = exp_data_current[well_name]['Temperature Data']

        # Remove background from chem data
        chem_data = chem_data - np.mean(chem_data[:3, :], axis=0)
        bounds_idx = time_to_index([tinitial, tfinal], time)
        
        # Remove inactive pixels from chem data
        chem_data = chem_data[:, idx_active]

        if visualise_steps:
            fig2, ax2 = plt.subplots(4, 1, figsize=(5, 3*4), dpi=300)
            fig2.suptitle(f'Processing. Experiment{experiment_id}', fontsize=22)

            ax2[0].plot(time, chem_data)
            ax2[0].set_xlabel('Time (s)', fontsize=18)
            ax2[0].set_ylabel('Voltage (mV)', fontsize=18)
            ax2[0].set_title('Step 1: Subtract backgound', fontsize=20)

        # Average active pixels
        chem_data_av = chem_data.mean(axis=1)
        temp_data_av = temp_data.mean(axis=1)

        if visualise_steps:
            ax2[1].plot(time, chem_data)
            ax2[1].plot(time, chem_data_av, c='k', linewidth=2, label='Mean')
            ax2[1].legend()
            ax2[1].set_xlabel('Time (s)', fontsize=18)
            ax2[1].set_ylabel('Voltage (mV)', fontsize=18)
            ax2[1].set_title('Step 2: Mean', fontsize=20)

        # Apply moving average filter (i.e. smoothing)
        MA_FILTER_RAW = 50
        chem_data_av_ma = filtfilt(b=np.ones(MA_FILTER_RAW) / MA_FILTER_RAW, a=[1], x=chem_data_av)

        if visualise_steps:
            ax2[2].plot(time, chem_data_av_ma, c='k', linewidth=2)
            ax2[2].set_xlabel('Time (s)', fontsize=18)
            ax2[2].set_ylabel('Voltage (mV)', fontsize=18)
            ax2[2].set_title('Step 3: Filter signal', fontsize=20)

        # DERIVATIVES ALGO
        # Calculate 1st derivative + smooth it
        chem_data_av_ma_diff_med_ma = myfilt(np.gradient(chem_data_av_ma))
        # Calculate 2nd derivative + smooth it
        chem_data_av_ma_diff2 = myfilt(np.gradient(chem_data_av_ma_diff_med_ma))

        # Find inflection points
        positive_infl_idx, negative_infl_idx = find_infl_points(chem_data_av_ma_diff2, time, tinitial, tfinal)

        if visualise_steps:
            ax2[3].plot(time, chem_data_av_ma_diff2, linewidth=2)
            ax2[3].set_title('Step 4-5: 2nd der + Infl points', fontsize=20)
            ax2[3].set_xlabel('Time (s)', fontsize=18)
            ax2[3].set_ylabel('2nd der', fontsize=18)
            for k, infl in enumerate(positive_infl_idx, 1):
                ax2[3].axvline(x=time[infl], color='r', linewidth=2)
            for k, infl in enumerate(negative_infl_idx, 1):
                ax2[3].axvline(x=time[infl], color='k', linewidth=2)
            plt.tight_layout()
            # plt.savefig('exp35_processing_wf.png')

        if idx_active.sum() == 0:
            positive = 'inconclusive'
        else:
            if len(positive_infl_idx) > 0:
                if len(negative_infl_idx) >= 5:
                    positive = 'inconclusive'
                else:
                    positive = 'positive'
            else:
                positive = 'negative'

        # Find time to positive: TTP is the first positive inflection point
        ttp = time[positive_infl_idx[0]] / 60 if (positive == 'positive') else 0

        # EXPERIMENT SUMMARY
        n_active_pixels = idx_active.sum()
        drift = abs(chem_data_av_ma[0] - chem_data_av_ma[bounds_idx[1]])
        ttp_str = f'{ttp:.2f}' if (positive == 'positive') else 'n/a'
        exp_data = {'Parameter': ['Experiment ID', 'Well', 'Active pixels', 'Drift (mV)', 'Result', 'TTP'],
                    'Data': [experiment_id, well_name, n_active_pixels, round(drift, 2), positive, ttp_str]}
        # Create DataFrame
        exp_df = pd.DataFrame(exp_data)

        # ALL EXPERIMENTS SUMMARY
        well_summary[well_name] = {'time': time,
                                   'average chem data': chem_data_av_ma,
                                   'average temp data': temp_data_av,
                                   'active pixels': n_active_pixels,
                                   'drift': drift,
                                   'result': positive,
                                   'ttp': ttp,
                                   'positive inflections idx': positive_infl_idx,
                                   'table df': exp_df}

        # PLOTS
        if visualise_plt:
            # Plot active/inactive pixels
            ax = axs[0]
            ax[i].imshow(idx_active.reshape(-1, 56), cmap='cividis')
            ax[i].set_title(f'{well_name} ACP', fontsize=20)

            ax = axs[1]
            if chem_data.shape[1] != 0:
                ax[i].plot(time, chem_data)
                ax[i].plot(time, np.mean(chem_data, axis=1), lw=2, c='k', label='mean')
                ax[i].axvspan(540, 1140, alpha=.5, color='yellow', label='9-19min')
                ax[i].set_title(f'{well_name} ACP BS signal', fontsize=20)
                ax[i].set_xlabel('Time (s)', fontsize=18)
                ax[i].set_ylabel('Voltage (mV)', fontsize=18)
                ax[i].legend(fontsize=16)

            # Plot data - smoothed data - 1st derivative
            ax = axs[2]
            ax[i].set_title(f'{well_name} ACP', fontsize=20)
            ax[i].set_xlabel('Time (s)', fontsize=18)
            ax[i].set_ylabel('Voltage (mV)', fontsize=18)
            ax[i].plot(time, chem_data_av, lw=2, label='Raw Data')
            ax[i].plot(time, chem_data_av_ma, lw=2,  label='Filtered Data')
            ax[i].legend(fontsize=16)

            # Plot 2nd derivative and inflection points
            ax = axs[3]
            ax[i].plot(time, chem_data_av_ma_diff2, lw=2, label='2nd derivative')
            ax[i].set_title(f'2nd der and Infl points', fontsize=20)
            ax[i].set_xlabel('Time (s)', fontsize=18)
            ax[i].set_ylabel('Voltage (mV)', fontsize=18)
            for k, infl in enumerate(positive_infl_idx, 1):
                ax[i].axvline(x=time[infl], color='r', label=f'PosInflection Point {k}')
            for k, infl in enumerate(negative_infl_idx, 1):
                ax[i].axvline(x=time[infl], color='k', label=f'Neg Inflection Point {k}')

            # Plot table of results
            ax = axs[4]
            ax[i].axis('tight')
            ax[i].axis('off')
            tab = ax[i].table(cellText=exp_df.values, colLabels=exp_df.keys(), loc='center')
            tab.set_fontsize(18)
            tab.scale(1.4, 1.4)

    if visualise_plt or visualise_steps:
        plt.tight_layout()
        if save_plt:
            plt.savefig(Path(exp_path, f'processing_{experiment_id}.eps'))
            print(f'Saved processing figure for {exp_path.stem}')
        plt.show()

    def final_result_logic(control_well, sample_well):
        if control_well['result'] != 'positive':
            return 'inconclusive'
        else:
            return sample_well['result']

    def compute_viral_load(final_result, sample_well):
        if final_result == 'positive':
            if sample_well['ttp'] <= 12.33:
                return 'low'
            elif sample_well['ttp'] <= 15.66:
                return 'medium'
            else:
                return 'high'
        elif final_result == 'negative':
            return 'no viral load'
        else:
            return 'inconclusive'

    exp_summary = {}
    exp_summary['experiment id'] = experiment_id
    exp_summary['final result'] = final_result_logic(well_summary['Top Well'], well_summary['Bot Well'])
    exp_summary['viral load'] = compute_viral_load(exp_summary['final result'], well_summary['Bot Well'])
    exp_summary['well data'] = well_summary

    return exp_summary


def algo(exp_path, tinitial=540, tfinal=1140, visualise_preprocessing=False, save_preprocessing=False, visualise_processing=False, save_processing=False, version='v2'):

    data = load_data(exp_path, version)
    exp_data_unprocessed = binary_to_wells(data, version)

    # for every time.. if time>tinitial
    exp_data_initial = get_initial_data(exp_data_unprocessed, version)
    exp_data_preproc = preprocessing_initial(exp_path, exp_data_initial, version)  # find settling time. filter by vref and vrange
    settled_idx_top = exp_data_preproc['Top Well']['Settled idx']
    settled_idx_bot = exp_data_preproc['Bot Well']['Settled idx']

    # print(f'Settling idx {settled_idx_top, settled_idx_bot}')

    tcurrent = tinitial+30
    exp_data_current = get_current_data(exp_data_unprocessed, settled_idx_top, settled_idx_bot, tcurrent)
    exp_data_preproc = preprocessing_update(exp_path, exp_data_current, exp_data_preproc, tinitial, visualise_plt=visualise_preprocessing, save_plt=save_preprocessing)

    tcurrent = tfinal
    exp_data_current = get_current_data(exp_data_unprocessed, settled_idx_top, settled_idx_bot, tcurrent)
    exp_summary = processing(exp_path, exp_data_current, exp_data_preproc, tinitial, tfinal, visualise_plt=visualise_processing, save_plt=save_processing)

    # for t<tinitial do noting - need to wait
    # fot the first t>tinitial, do preprocessing initial

    # for all the others, do update.

    # fot t > tfinal do nothing

    return exp_summary


if __name__ == "__main__":
    # exp_path = Path('..', 'data_files', '190520_2_14')
    # exp_path = Path('..', 'data_files', '180520_6_35')
    # exp_path = Path('..', 'data_files', '250520_7_64')
    # exp_path = Path('..', 'data_files', '150520_4_2_86')
    exp_path = Path('..', 'data_files', '250520_2_97')
    # exp_path = Path('..', 'data_files', '150520_2_118')
    # exp_path = Path('..', 'data_files', '150520_5_129')
    # exp_path = Path('..', 'data_files', '180520_4_165')
    # exp_path = Path('..', 'data_files', '120520_4_10x_155')
    # exp_path = Path('..', 'data_files_v3', 'D20210319_E00_C00_F4500KHz_U_ST44_DNA')
    # exp_path = Path('..', 'data_files_v3', 'D20210319_E00_C00_F4500KHz_U_ST45_DNA')
    out = algo(exp_path, visualise_preprocessing=False, save_preprocessing=False, visualise_processing=False, save_processing=False, version='v2')
    for well_name, well_data in out['well data'].items():
        print(well_data['table df'], end='\n')

    # curr_path = Path('..', 'data_files')
    # experiments = [x for x in curr_path.iterdir() if x.is_dir()]
    # for exp_path in tqdm(experiments):
    #     out = algo(exp_path, visualise_preprocessing=False, save_preprocessing=False, visualise_processing=True,
    #                save_processing=False)
    #     # for well_name, well_data in out['well data'].items():
    #         # print(well_data['table df'], end='\n')
    #     print('---')
