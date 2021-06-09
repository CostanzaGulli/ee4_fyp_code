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

version = 'v2'

def binary_file_read(file):
    fw = open(file, 'rb')
    data_byte = fw.read()
    data_len = len(data_byte) >> 1
    data_list = []
    for n in range(0, data_len):
        (data, ) = struct.unpack('>H', data_byte[2*n:2*(n+1)])
        data_list.append(data)
    return data_list


def load_data(exp_path):
    if version == 'v4':
        filename = [i for i in exp_path.glob("VF*.bin")]
    elif version == 'v2':
        filename = [i for i in exp_path.glob("*.bin")]
    else:
        raise NotImplemented(f'The code for version {version} of the chip has not been implemented')

    if len(filename) > 0:
        filename = filename[0]
    else:
        print(f'No bin file found in {exp_path}')

    data = binary_file_read(filename)
    return data


def binary_to_numpy(data, f_start, f_end):
    ''' Convert list of data to numpy representation

    Parameters
    ----------
    data : list
        list of input data to be converted into numpy array
    f_start : int
        first frame to be considered
    f_end : int
        last frame to be considered. To set it change code. Now it's len(data)//4372

    Returns
    -------
    tuple
        (time_vect, frame_3d, temp_vect) where
        - time_vect : ndarray
            1D array containing the sampling times
        - frame_3d : ndarray
            3D array containing time frames of the sensor array outputs
            Dimensions 0 and 1 are rows and columns of the physical array
            Dimension 2 is time
        - temp_vect : ndarray
            1D array
    '''
    # convert data into 3D object [x, y, time]
    #f_end = len(data)//4372  # remove line when we move to GUI
    if version =='v2':
        N = 4372
    elif version == 'v4':
        N = 4374
    else:
        raise NotImplemented(f'The code for version {version} of the chip has not been implemented')

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
    # print(f'time_vect.shape, frame_3d.shape, temp_vect.shape {time_vect.shape, frame_3d.shape, temp_vect.shape}')
    return time_vect, frame_3d, temp_vect


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
    bot_well = frame_3d[-39:, :, :].reshape(-1, n_time, order='C').T # obtain the bottom well
    # and reshape the 3D matrix N'' x M x T to a 2D matrix of dimensions T x N''M (N' + N'' = N)

    return top_well, bot_well


def binary_to_wells(data):
    # Get data from the bin file. Split chem/temp and top/bottom wells
    f_start, f_end = 1, 976  # This is set by Lei somewhere
    time_vect, frame_3d, temp_vect = binary_to_numpy(data, f_start, f_end)  # build data arrays from the binary file

    n_rows, n_cols, n_time = frame_3d.shape  # find data dimensions: TxNxM = samples x rows of sensors x columns
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


def get_initial_data(exp_data_unprocessed):
    time_vect = exp_data_unprocessed['Top Well']['Time']
    top_well = exp_data_unprocessed['Top Well']['Chemical Data']
    bot_well = exp_data_unprocessed['Bot Well']['Chemical Data']
    top_well_temp = exp_data_unprocessed['Top Well']['Temperature Data']
    bot_well_temp = exp_data_unprocessed['Bot Well']['Temperature Data']

    # fig, ax = plt.subplots(figsize=(12, 3))
    # ax.plot(time_vect, top_well)
    # ax.set_title('Original data time series. Experiment 35', fontsize=20)
    # ax.set_xlabel('Time (s)', fontsize=18)
    # ax.set_ylabel('Voltage (mV)', fontsize=18)
    # # plt.savefig('exp35_originaldata.png')
    # plt.show()

    if version == 'v2':
        tstart_idx = time_to_index([930], time_vect)[0]
    elif version =='v4':
        tstart_idx = time_to_index([900], time_vect)[0]  # TODO not 900. what?
    else:
        raise NotImplemented(f'The code for version {version} of the chip has not been implemented')

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
        1D array of bool with dimension (NxM). For each pixel, returns True if, during the first 10 samples,
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

# Identify active pixels by checking that the absolute value of the derivative is always below 10 (jumps are noise)
def filter_by_derivative(X, thresh=5):
    x_diff = np.abs(np.diff(X, axis=0))
    return (x_diff < thresh).all(axis=0)


def time_to_index(times, time_vect):
    '''
    Returns index of the times closest to the desired ones

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
    '''

    Parameters
    ----------
    time_vect : ndarray
        1D array with dimension T containing the sampling times
    X : ndarray
        2D array with dimension TxNM containing the sampled data
    bounds : (int, int), optional
        tuple containing the minimum and maximum times (in ms) where the loading time has to be searched.
        Default is (600, 900)
    viz : bool, optional
        if viz=True, show the plot. Default is False

    Returns
    -------
    loading_index : int
        T index at which the loading occurs
    loading_time : int
        time at which the loading occurs
    '''

    search_start, search_end = time_to_index(bounds, time_vect)  # for each time in bounds, find the index
    print(f'bounds {search_start, search_end}')
    # of the sample (in time_vect) that is closest to the desired one (in bounds)
    X_mean = np.mean(X, axis=1)  # for each sample, calculate the mean of all pixels
    X_mean_diff = np.diff(X_mean)  # find the derivative

    loading_index = np.argmax(X_mean_diff[search_start:search_end]) + search_start + 1  # find the index
    # where the derivative is max in the specified interval
    loading_index = loading_index + 10  # add settling time
    loading_time = time_vect[loading_index]  # find the time that index corresponds to

    if viz:  # if viz is true, plot the following
        fig, ax = plt.subplots(3, 1)
        fig.suptitle('Finding Loading Time...')

        ax[0].set(title='Active Chemical Pixels, ACP')
        ax[0].plot(time_vect, X)  # plot the active chemical pixels

        ax[1].set(title='Mean(ACP)')
        ax[1].plot(time_vect, X_mean)  # plot the average of the pixels
        ax[1].axvline(time_vect[search_start], color='C1')  # plot vertical line: beginning of the interval
        ax[1].axvline(time_vect[search_end], color='C1')  # plot vertical line: end of the interval
        ax[1].axvline(loading_time, color='C2')  # plot vertical line: the loading time that was found

        ax[2].set(title='Diff(Mean(ACP))')
        ax[2].plot(time_vect[1:], X_mean_diff)  # plot the derivative of the mean
        ax[2].axvline(time_vect[search_start], color='C1')  # plot vertical line: beginning of the interval
        ax[2].axvline(time_vect[search_end], color='C1')  # plot vertical line: end of the interval
        ax[2].axvline(loading_time, color='C2')  # plot vertical line: the loading time that was found

        plt.tight_layout()
        plt.show()
    return loading_index, loading_time


def find_settled_time(time_vect, temp):  # for v4
    temp_mean = np.mean(temp, axis=1)
    threshold = temp_mean[0]+0.95*(temp_mean[-1]-temp_mean[0])
    settled_idx = np.argmax(temp_mean > threshold)
    return settled_idx, time_vect[settled_idx]  # return the index and the time that corresponds to


def split_chem_and_temp(arr):
    """ Separates temperature and chemical pixels.

    Parameters
    ----------
    arr : ndarray
        3D array of chemical and temperature pixels. There is 1 temp pixel for each 8 chem pixels

    Returns
    -------
    arr_temp, arr_chem where
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


def decaying_exp_neg(x, b):
    """ a: min, b: slope, c: c shift along x axis, d: value at x = c """
    return np.exp(-b * x)


def decaying_exp_pos(x, b):
    return 1-np.exp(-b * x)


def decaying_exp(x, a, b):
    return a*(1-np.exp(-b * x))


def fit_pixels_interpolate(time, X, idx_active, interpolate_idx, drift_idx, popt0=[-10, 0.1]):
    # popt = np.array(np.zeros(X.shape[1]))  # modify for fitting with more parameters
    popt = np.zeros((2, X.shape[1]))

    # for every pixel
    for i in range(X.shape[1]):
        if idx_active[i]:  # if the pixel is active...
            data = filtfilt(b=np.ones(10) / 10, a=[1], x=X[:, i])

            # Fit the curve (interpolate)
            try:
                popt[:, i], pcov = curve_fit(decaying_exp, time[:interpolate_idx], data[:interpolate_idx], p0=[-10, 0.1])

                # data_scaled_fit = decaying_exp(time[:interpolate_idx], *popt[:, i])
                # fig, ax = plt.subplots()
                # ax.plot(time[:interpolate_idx], data[:interpolate_idx])
                # ax.plot(time[:interpolate_idx], data_scaled_fit)
                # ax.set_title(f'fitting with parameters {popt[:, i]}')
                # plt.show()
            except:
                print('EXCEPT: could not fit this pixel')
                popt[:, i] = 100

                # print(popt[:, i])
                # fig, ax = plt.subplots()
                # ax.plot(time, data)
                # plt.show()
        else:
            popt[:, i] = 100
    return popt


def fit_pixels_extrapolate(time, X, idx_active, extrapolate_idx, drift_idx, popt, thresh_active=12): 
    idx_active_fit = np.array(np.zeros(X.shape[1]), dtype=bool)
    fit_error = np.array(np.zeros((extrapolate_idx, X.shape[1])))

    # for every pixel
    for i in range(X.shape[1]):
        if idx_active[i]:  # if the pixel is active...
            data = filtfilt(b=np.ones(10) / 10, a=[1], x=X[:, i])
            drift = data[drift_idx] - data[0]
            drift_magnitude = np.abs(drift)  # Find drift magnitude (in V)

            # Fit the curve (extrapolate)
            # if drift_magnitude != 0 and popt[i] < 100:
            if drift_magnitude != 0 and popt[1, i] < 100:
                data_fit = decaying_exp(time[:extrapolate_idx], *popt[:,i])
                fit_error[:, i] = np.abs(data[:extrapolate_idx] - data_fit)
                idx_active_fit[i] = (fit_error[:, i] < thresh_active).all()

                # fig, ax = plt.subplots()
                # ax.plot(time[:extrapolate_idx], data[:extrapolate_idx])
                # ax.plot(time[:extrapolate_idx], data_fit)
                # plt.show()
            else:
                idx_active_fit[i] = False
        else:
            idx_active_fit[i] = False
    idx_active = idx_active & idx_active_fit
    fit_error[:, ~idx_active] = 0
    return idx_active, fit_error


def preprocessing_initial(exp_path, exp_data_initial, visualise_filtering=False):
    time_vect = exp_data_initial['Top Well']['Time']
    top_well = exp_data_initial['Top Well']['Chemical Data']
    bot_well = exp_data_initial['Bot Well']['Chemical Data']
    top_well_temp = exp_data_initial['Top Well']['Temperature Data']
    bot_well_temp = exp_data_initial['Bot Well']['Temperature Data']

    # find_active_pixels = lambda x: filter_by_vref(x) & filter_by_vrange(x)
    # filter_by_vref finds pixels that are in contact with the solution
    # filter_by_vrange finds the pixels that have values always within a certain range,
    #   i.e. their value does not decay nor saturate
    # here, define find_active_pixels to find the pixels that satisfy both of the above conditions
    # idx_top_active, idx_bot_active = [find_active_pixels(i) for i in [top_well, bot_well]]
    # find active pixels for both the top and bottom well

    idx_top_active_vref = filter_by_vref(top_well)
    idx_bot_active_vref = filter_by_vref(bot_well)
    idx_top_active_vrange = filter_by_vrange(top_well)
    idx_bot_active_vrange = filter_by_vrange(bot_well)
    idx_top_active = idx_top_active_vref & idx_top_active_vrange
    idx_bot_active = idx_bot_active_vref & idx_bot_active_vrange

    if visualise_filtering:
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

    idx_top_active, idx_bot_active = cleanup_pixels(idx_top_active), cleanup_pixels(idx_bot_active)
    # for both the top and bottom well, find the largest active region

    if visualise_filtering:
        ax[2, 0].imshow(idx_top_active.reshape(-1, 56), cmap='cividis')
        ax[2, 0].set_title('Step 3: Filtered spatially', fontsize=20)

        ax[2, 1].plot(time_vect, top_well[:, idx_top_active])
        ax[2, 1].set_xlabel('Time (s)', fontsize=18)
        ax[2, 1].set_ylabel('Voltage (mV)', fontsize=18)
        plt.tight_layout()
        # plt.savefig('exp64_preproc_initial.eps')
        plt.show()

    assert idx_top_active.sum() != 0, f'No active pixels in the top well. Experiment {exp_path.stem} invalid.'
    assert idx_bot_active.sum() != 0, f'No active pixels in the bot well. Experiment {exp_path.stem} invalid.'

    if version == 'v2':
        settled_idx_top, settled_t_top = find_loading_time(time_vect, top_well[:, idx_top_active], bounds=(600, 900))
        settled_idx_bot, settled_t_bot = find_loading_time(time_vect, bot_well[:, idx_bot_active], bounds=(600, 900))
    elif version == 'v4':
        settled_idx_top, settled_t_top = find_settled_time(time_vect, top_well_temp)
        # top well: find how long it takes for the signal to settle after the chip was loaded (sample index and time)
        settled_idx_bot, settled_t_bot = find_settled_time(time_vect, bot_well_temp)
        # bottom well: find how long it takes for the signal to settle after the chip was loaded (sample index and time)
    else:
        raise NotImplemented(f'The code for version {version} of the chip has not been implemented')


    if visualise_filtering:
        fig, ax = plt.subplots(figsize=(12, 3), dpi=300)
        ax.plot(time_vect, top_well)
        ax.axvline(x=time_vect[settled_idx_top-10], color='k', linestyle='--', linewidth=2, label='Loading time')
        ax.axvline(x=time_vect[settled_idx_top], color='k', linewidth=2, label='Settled time')
        ax.set_title(f'Original data time series (until t_start). Experiment {experiment_id}', fontsize=20)
        ax.set_xlabel('Time (s)', fontsize=18)
        ax.set_ylabel('Voltage (mV)', fontsize=18)
        ax.legend(fontsize=18, bbox_to_anchor=(1, 1))
        # plt.savefig('exp64_data_preproc_initial.eps')
        plt.show()

    x_top = time_vect[settled_idx_top:] - time_vect[settled_idx_top]
    x_bot = time_vect[settled_idx_bot:] - time_vect[settled_idx_bot]
    Y_top = top_well[settled_idx_top:, :]
    Y_bot = bot_well[settled_idx_bot:, :]
    temp_top, temp_bot = top_well_temp[settled_idx_top:], bot_well_temp[settled_idx_bot:]  # temperature data after settling

    # Save data for the experiment
    exp_data = OrderedDict({  # TODO here only need time, settled idx and active pixels
        'Top Well': {
            'Time': x_top,
            'Chemical Data': Y_top,
            'Active Pixels': idx_top_active,
            'Temperature Data': temp_top,
            'Settled idx': settled_idx_top
        },

        'Bot Well': {
            'Time': x_bot,
            'Chemical Data': Y_bot,
            'Active Pixels': idx_bot_active,
            'Temperature Data': temp_bot,
            'Settled idx': settled_idx_bot
        }
    })
    return exp_data


def preprocessing_update(exp_path, exp_data_current, exp_data_preproc, tinitial, visualise_plt=False, save_plt=False, visualise_filtering=True):
    f_start, f_end = 1, 976  # This is set by Lei somewhere

    idx_top_active = exp_data_preproc['Top Well']['Active Pixels']
    idx_bot_active = exp_data_preproc['Bot Well']['Active Pixels']
    settled_idx_top = exp_data_preproc['Top Well']['Settled idx']
    settled_idx_bot = exp_data_preproc['Bot Well']['Settled idx']

    x_top = exp_data_current['Top Well']['Time']
    x_bot = exp_data_current['Bot Well']['Time']
    Y_top = exp_data_current['Top Well']['Chemical Data']
    Y_bot = exp_data_current['Bot Well']['Chemical Data']
    temp_top = exp_data_current['Top Well']['Temperature Data']
    temp_bot = exp_data_current['Bot Well']['Temperature Data']

    if visualise_filtering:
        fig, ax = plt.subplots(4, 2, figsize=(10, 3*4))
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

    if visualise_filtering:
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

    if visualise_filtering:
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
    print(np.isnan(Y_bot_bs).sum())

    # Interpolate signal fot fitting
    # XX changed the bs signal
    popt_top = fit_pixels_interpolate(x_top, Y_top_bs, idx_top_active, tinitial_idx_top, tcurrent_idx_top-1, popt0=None)
    popt_bot = fit_pixels_interpolate(x_bot, Y_bot_bs, idx_bot_active, tinitial_idx_bot, tcurrent_idx_bot-1, popt0=None)

    # Filter pixels by k-means clustering
    if idx_top_active.sum() > 6:
        idx_top_active = cleanup_by_kmeans(Y_top_bs, idx_top_active)
    # print(bot_well_settled.shape, idx_bot_active.shape)
    if idx_bot_active.sum() > 6:
        idx_bot_active = cleanup_by_kmeans(Y_bot_bs, idx_bot_active)

    if visualise_filtering:
        ax[3, 0].imshow(idx_top_active.reshape(-1, 56), cmap='cividis')
        ax[3, 0].set_title('Step 3: Filtered by kmeans', fontsize=20)

        ax[3, 1].plot(x_top, Y_top_bs[:, idx_top_active])
        ax[3, 1].set_xlabel('Time (s)', fontsize=18)
        ax[3, 1].set_ylabel('Voltage (mV)', fontsize=18)
        plt.tight_layout()
        # plt.savefig('exp35_preproc_update.eps')
        plt.show()

    # Further cleanup of pixels  # TODO where does this go?
    # idx_top_active, idx_bot_active = cleanup_pixels(idx_top_active), cleanup_pixels(idx_bot_active)

    if idx_top_active.sum() == 0:
        print(f'No active pixels in the top well. Experiment {exp_path.stem} invalid.')  # TODO make it some kind of assertion error
    if idx_bot_active.sum() == 0:
        print(f'No active pixels in the bottom well. Experiment {exp_path.stem} invalid.')

    # Save data for the experiment
    exp_data_preproc = OrderedDict({
        'Top Well': {
            'Time': x_top,
            'Chemical Data': Y_top,
            'Active Pixels': idx_top_active,
            'Temperature Data': temp_top,
            'Fitting Parameters': popt_top,
            'Settled idx': settled_idx_top
        },

        'Bot Well': {
            'Time': x_bot,
            'Chemical Data': Y_bot,
            'Active Pixels': idx_bot_active,
            'Temperature Data': temp_bot,
            'Fitting Parameters': popt_bot,
            'Settled idx': settled_idx_bot
        }
    })

    if visualise_plt:
        N_axes = 6  # plots for the top and bottom well
        fig, ax = plt.subplots(N_axes, 2, figsize=(10, 3 * N_axes))
        fig.suptitle(f"Experiment Name: {exp_path.stem}")
        ax_top, ax_bot = ax[:, 0], ax[:, 1]

        for ax, x, Y, idx_active, temp, label in [(ax_top, x_top, Y_top[:, idx_top_active], idx_top_active, temp_top, 'Top'),
                                                  (ax_bot, x_bot, Y_bot[:, idx_bot_active], idx_bot_active, temp_bot, 'Bot')]:

            ax[0].set_title(f'{label} Well')
            ax[0].imshow(idx_active.reshape(-1, 56), cmap='cividis')  # plot active/inactive pixels

            ax[1].set(title='Temp Pixels, TP')
            print(f'X DOPO {x.shape}')

            ax[1].plot(x, temp)  # plot temperature in time

            ax[2].set(title='Active Chemical Pixels, ACP')
            if Y.shape[1] > 0:
                ax[2].plot(x, Y)  # plot chem pixels in time if there are active pixels

            Y_bs = Y - np.mean(Y[:3, :], axis=0)  # remove the background by subtracting the first value
            ax[3].set(title='Background-subtracted ACP')
            if Y_bs.shape[1] > 0:
                ax[3].plot(x, Y_bs)
                ax[3].plot(x, np.mean(Y_bs, axis=1), lw=2, color="k", label='Mean')  # plot the the signal with no offset
                ax[3].legend()

            Y_bs_smooth = pd.DataFrame(Y_bs).rolling(30).mean().values  # filter the data with a MA(30)
            # i.e. to smooth out the signal, each value is the average of the 30 surrounding ones
            Y_bs_smooth_mean = np.mean(Y_bs_smooth, axis=1)
            ax[4].set(title='MA(30) filtered ACF')
            if Y_bs.shape[1] > 0:
                ax[4].plot(x, Y_bs_smooth)  # plot the smoothed signal
                ax[4].plot(x, Y_bs_smooth_mean, lw=2, color="k", label='Mean')
                ax[4].legend()

            # Y_bs_smooth_diff = np.diff(Y_bs_smooth, axis=0)  # compute the derivative of the signal where
            # # Y_bs_smooth_diff(i) = Y_bs_smooth(i+1) - Y_bs_smooth(i)
            # ax[5].set(title='1st derivative')
            # ax[5].plot(x[1:], Y_bs_smooth_diff)
            # ax[5].plot(x[1:], np.mean(Y_bs_smooth_diff, axis=1), lw=2, color="k", label='Mean')  # plot derivative
            # ax[5].legend()

            ax[1].get_shared_x_axes().join(*ax[1:5])

            ax[5].set(title='Average fit error')
            bounds = time_to_index([540, 1140], x)  # average error between 9 and 19min (where positive can occurr)
            # pos = ax[5].imshow(np.mean(fit_error[bounds[0]:bounds[1]], axis=0).reshape(-1, 56), cmap='cividis')
            # fig.colorbar(pos, ax=ax[5])

        plt.tight_layout()
        if save_plt:
            plt.savefig(Path(exp_path, 'preprocessing.eps'))
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


def processing(exp_path, exp_data_preproc, tinitial, tfinal, visualise_plt=False, save_plt=False):

    # Get string of experiment ID
    exp_path_str = exp_path.stem
    experiment_id = exp_path_str[exp_path_str.rfind('_') + 1:]
    n_wells = len(exp_data_preproc)

    well_summary = {}

    if visualise_plt:
        fig, axs = plt.subplots(7, n_wells, figsize=(10, 7 * 3), dpi=100)
        fig.suptitle(f'Experiment: {exp_path_str}')

    # For every well...
    for i, (well_name, well) in enumerate(exp_data_preproc.items()):
        # Get preprocessing data
        time = well['Time']
        idx_active = well['Active Pixels']
        chem_data = well['Chemical Data']
        temp_data = well['Temperature Data']
        popt = well['Fitting Parameters']

        # Remove background from chem data
        chem_data = chem_data - np.mean(chem_data[:3, :], axis=0)

        # Extrapolate from coefficients + find the error compared to the real curve and identify inactive pixels
        idx_active, fit_error = fit_pixels_extrapolate(time, chem_data, idx_active, chem_data.shape[0], chem_data.shape[0]-1, popt)
        
        # Remove inactive pixels from chem data
        chem_data = chem_data[:, idx_active]

        bounds_idx = time_to_index([tinitial, tfinal], time)  # todo: dont need end bound because dont do processing there

        # Average active pixels
        chem_data_av = chem_data.mean(axis=1)  # TODO: ERROR "MEAN OF EMPTY SLICE"
        temp_data_av = temp_data.mean(axis=1)

        # Apply moving average filter (i.e. smoothing)
        MA_FILTER_RAW = 50  # TODO WAS 155
        chem_data_av_ma = filtfilt(b=np.ones(MA_FILTER_RAW) / MA_FILTER_RAW, a=[1], x=chem_data_av)

        # DERIVATIVES ALGO
        # Calculate 1st derivative + smooth it
        chem_data_av_ma_diff_med_ma = myfilt(np.gradient(chem_data_av_ma))
        # Calculate 2nd derivative + smooth it
        chem_data_av_ma_diff2 = myfilt(np.gradient(chem_data_av_ma_diff_med_ma))

        # Find inflection points
        positive_infl_idx, negative_infl_idx = find_infl_points(chem_data_av_ma_diff2, time, tinitial, tfinal)

        if len(positive_infl_idx) > 0:
            if len(negative_infl_idx) >= 5:
                positive = 'inconclusive'
            else:
                positive = 'positive'
        else:
            positive = 'negative'

        # Find time to positive: TTP is the first positive inflection point
        ttp = time[positive_infl_idx[0]] / 60 if (positive == 'positive') else 0

        # FITTING ALGO
        # TODO THE SMALLEst BETWEEN TIME1 AND TCURRENT
        max_fit_err_at_infl = -2
        for j, k in enumerate(positive_infl_idx):
            fit_err_at_infl = np.mean(fit_error[k, idx_active])
            if fit_err_at_infl > max_fit_err_at_infl:
                max_fit_err_at_infl = fit_err_at_infl
        max_fit_error = np.amax(fit_error[bounds_idx[0]:bounds_idx[1], :], axis=0)
        print(f'MAX FIT ERROR {max_fit_err_at_infl}')

        # Find number of pixels with error above threshold for positivity
        thresh_positive = 6
        pos_count = (max_fit_error > thresh_positive).sum()
        if idx_active.sum() != 0:
            pos_percentage = pos_count / idx_active.sum()
        else:
            pos_percentage = 100000  # TODO changee
        # print(f'number pixels above threshold for positivity = {pos_count}')
        # print(f'% of active pixels above threshold for positivity = {pos_percentage}')

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
                                   'average chem data': chem_data_av_ma,  # TODO need to store all the data to reuse code!!
                                   'average temp data': temp_data_av,
                                   'active pixels': n_active_pixels,
                                   'drift': drift,
                                   'result': positive,
                                   'ttp': ttp,
                                   'positive inflections idx': positive_infl_idx,
                                   'pos count': pos_count,
                                   'pos percentage': pos_percentage,  # TODO added these two. needed?
                                   'max fit error at infl': max_fit_err_at_infl,
                                   'table df': exp_df}

        # PLOTS
        if visualise_plt:
            # Plot active/inactive pixels
            ax = axs[0]
            ax[i].imshow(idx_active.reshape(-1, 56), cmap='cividis')
            ax[i].set(title=f'{well_name} ACP')

            # Plot data - smoothed data - 1st derivative
            ax = axs[1]
            ax[i].set(title=f'{well_name}',
                      xlabel='Time (s)',
                      ylabel='Voltage (mV)')
            ax[i].plot(time, chem_data_av, label='Raw Data')
            ax[i].plot(time, chem_data_av_ma, label='Smooth Data')
            ax2 = ax[i].twinx()
            ax2.plot(time, chem_data_av_ma_diff_med_ma, label='1st Derivative', color='C3')
            ax[i].legend()
            ax2.legend()

            # Plot 2nd derivative and inflection points
            ax = axs[2]
            ax[i].plot(time, chem_data_av_ma_diff2, label='2nd derivative')
            ax[i].set(title='2nd Derivative and Inflection Points',
                      xlabel='Time (s)',
                      ylabel='(mV)')
            for k, infl in enumerate(positive_infl_idx, 1):
                ax[i].axvline(x=time[infl], color='r', label=f'PosInflection Point {k}')
            for k, infl in enumerate(negative_infl_idx, 1):
                ax[i].axvline(x=time[infl], color='k', label=f'Neg Inflection Point {k}')

            # Plot table of results
            ax = axs[3]
            ax[i].axis('tight')
            ax[i].axis('off')
            table = pd.plotting.table(ax[i], exp_df, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(18)
            table.scale(1.3, 1.3)

            ax = axs[4]
            if chem_data.shape[1] != 0:
                ax[i].plot(time, chem_data)
                ax[i].plot(time, np.mean(chem_data, axis=1), lw=2, c='k', label='mean')
                ax[i].axvspan(540, 1140, alpha=.5, color='yellow', label='9-19min')
                ax[i].set_title('BS average signal (active pixels)')
                ax[i].legend()

            # plot max error in the 9-19 min range for each pixels
            ax = axs[5]
            pos = ax[i].imshow(max_fit_error.reshape(-1, 56), cmap='cividis')
            ax[i].set_title('Max fit error (9-19min)')
            fig.colorbar(pos, ax=ax[i])

            # plot average error of fit for the active pixels
            ax = axs[6]
            ax[i].plot(time, np.mean(fit_error[:, idx_active], axis=1), label='average error')
            ax[i].axvspan(540, 1140, alpha=.5, color='yellow', label='9-19min')
            ax[i].set_title('Average error (active pixels)')
            ax[i].legend()

    if visualise_plt:
        plt.tight_layout()
        if save_plt:
            plt.savefig(Path(exp_path, 'processing.eps'))
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


def algo(exp_path, tinitial=540, tfinal=1140, visualise_preprocessing=False, save_preprocessing=False, visualise_processing=False, save_processing=False):
    data = load_data(exp_path)

    exp_data_unprocessed = binary_to_wells(data)
    # for every time.. if time>tinitial
    exp_data_initial = get_initial_data(exp_data_unprocessed)
    exp_data_preproc = preprocessing_initial(exp_path, exp_data_initial)  # find settling time. filter by vref and vrange

    settled_idx_top = exp_data_preproc['Top Well']['Settled idx']
    settled_idx_bot = exp_data_preproc['Bot Well']['Settled idx']

    print(f'Settling idx {settled_idx_top, settled_idx_bot}')

    # TODO: have the possiblity of only processing 1 well or the other when 1 is active and the other not yet/not anymore
    # TODO: splot fit_expo into finding coefficients and modelling. need to do that also in the preprocessing when the processing was not on the same data.

    tcurrent = tfinal
    exp_data_current = get_current_data(exp_data_unprocessed, settled_idx_top, settled_idx_bot, tcurrent)
    exp_data_preproc = preprocessing_update(exp_path, exp_data_current, exp_data_preproc, tinitial, visualise_plt=visualise_preprocessing, save_plt=save_preprocessing)
    exp_summary = processing(exp_path, exp_data_preproc, tinitial, tfinal, visualise_plt=visualise_processing, save_plt=save_processing)

    # for t<tinitial do noting - need to wait
    # fot the first t>tinitial, do preprocessing initial

    # for all the others, do update. TODO: processing division between first and update.

    # fot t > tfinal do nothing
    return exp_summary


if __name__ == "__main__":
    # exp_path = Path('.', '210520_6_27')
    exp_path = Path('..', 'data_files', '250520_7_64')
    # exp_path = Path('..', 'data_files', '210520_6_27')
    # exp_path = Path('..', 'data_files_v3', 'D20210316_E00_C00_F4500KHz_U_ST28_DNA')
    out = algo(exp_path, visualise_preprocessing=False, save_preprocessing=False, visualise_processing=False, save_processing=False)
    #for well_name, well_data in out['well data'].items():
    #    print(well_data['table df'], end='\n')

    # curr_path = Path('..', 'data_files')
    # experiments = [x for x in curr_path.iterdir() if x.is_dir()]
    # for exp_path in tqdm(experiments):
    #     out = algo(exp_path, visualise_preprocessing=False, save_preprocessing=False, visualise_processing=False,
    #                save_processing=False)
    #     # for well_name, well_data in out['well data'].items():
    #         # print(well_data['table df'], end='\n')
    #     print('---')
