"""

Refactored version of job_run.py with multiprocessing.
We use multiprocessing here because Python doesn't implement multithreading.

"""

from __future__ import division

from multiprocessing import Process, Manager
from time import time
from skimage import io
from piv_tools import contrast
from datetime import datetime

# TODO: We should really only import functions/classes that we need instead of using wildcard

import numpy as np
import os
import functools
import json

import glob
import sys

# import matplotlib.pyplot as plt
import scipy.interpolate as interp

if sys.version_info[0] == 3:
    print("This system is using Python 3, but OpenPIV requires Python 2")

# OpenPIV requires Python 2
if sys.version_info[0] == 2:
    import openpiv.filters

# ===================================================================================================================
# WARNING: File read/close is UNSAFE in multiprocessing applications because multiple
# threads are accessing &/or writing to the same file. Please remember to use a queue if doing file I/O concurrently
# ===================================================================================================================

# Timing decorator (function_type being I/O or compute)
# Note: Must unpack both the dict and the value in this function

# To track distribution of runtime in functions using decorators
manager = Manager()
runtime_queue = manager.Queue()

def measure_runtime_arg(queue, function_type):
    def measure_runtime(func):
        @functools.wraps(func) # preserve introspection of wrapped function (now func.__name__ returns func_name and not 'wrap')
        def wrap(*args, **kwargs):
            start_time = time()
            res = func(*args, **kwargs)
            finish_time = time()

            func_name = func.__name__
            func_runtime = finish_time - start_time

            # Record relevant information: function name, function time, function type
            queue.put({'func_name': func_name, 'func_runtime': func_runtime, 'func_type': function_type})
            return res
        return wrap
    return measure_runtime


def aggregate_runtime_metrics(queue, num_images, num_processes, filename):
    # In: multiprocess queue, number of images, number of processes, file to write to
    # Out: summary object for possible use in plots/graphs/figures

    # total_dict will look like the following after tracking elements in the queue:
    # {'func_name': {'total_avg/io_avg': <float>,
    #                'longest': <int>,
    #                'total/io': <int>},
    #  'pCompute': <float>,
    #  'pIO': <float>
    # }
    total_dict = {}
    # Toggle this to print each entry into a file, otherwise only add aggregate lines
    print_all = False

    with open(filename, 'a+') as file:
        # Leave a timestamp & ASCII section header
        section_header = "========================================\n" + \
                         str(datetime.utcnow()) + \
                         "\n========================================\n"

        file.write(section_header)

        while not queue.empty():
            element = queue.get()
            
            if isinstance(element, str):
                file.write(element)
                continue

            el_func_name = element['func_name']
            ftype = element['func_type']
            runtime = element['func_runtime']
            
            if el_func_name in total_dict:
                current_func = total_dict[el_func_name]

                if ftype in current_func:
                    current_func[ftype] += runtime
                else:
                    current_func[ftype] = runtime

                if runtime > current_func['longest']:
                    current_func['longest'] = runtime
            else:
                total_dict[el_func_name] = {ftype: runtime, 'longest': runtime}

            if print_all:
                line_string = 'Function: %s, Type: %s, Runtime: %d \n' % (el_func_name, ftype, runtime)
                file.write(line_string)

        agg = {'total_time': 0, 'io_time': 0}

        # Dump raw data into separate file to be post-processed for graphs after interpretation
        summary = {}

        for key in total_dict:
            current_func = total_dict[key]
            longest = current_func['longest']

            if key == 'save_files':
                ftype = 'io'
                total = current_func['io']
                current_func['io_avg'] = total / (num_processes*8)
                agg['io_time'] += total
            else:  
                ftype = 'total'
                total = current_func['total']
                current_func['total_avg'] = total / num_processes
                agg['total_time'] += total

            average_per_process = current_func[ftype + '_avg']
            average_per_image = total/num_images # This metric doesn't mean much if there's more than 1 process
            average_per_process_per_image = average_per_process/num_images

            line_string = 'Function: %-25s, ' \
                          'Type: %-15s, ' \
                          'Average Per Process: %-10f, ' \
                          'Average Per Image: %-10f,' \
                          'Average PPPI: %-10f,' \
                          'Longest: %-5f \n' \
                          % (key, ftype,
                             average_per_process,
                             average_per_image,
                             average_per_process_per_image,
                             longest)
            file.write(line_string)

            summary[key] = {'type': ftype,
                            'APP': average_per_process,
                            'API': average_per_image,
                            'APPPI': average_per_process_per_image,
                            'longest': longest}

        pIO = agg['io_time']/agg['total_time']*100
        pCompute = 100 - pIO

        total_string = 'Percent Time Computing: %f, Percent Time Reading/Writing Files: %f\n' % (pCompute, pIO)
        file.write(total_string)

        with open('runtime_metric_obj.json', 'a+') as obj_file:
            # note: when using shell to access this data, use json.load
            json.dump(summary, obj_file)
            obj_file.write('\n') # so we can parse line by line later


# ===================================================================================================================
# MULTIPROCESSING UTILITY CLASSES & FUNCTIONS
# ===================================================================================================================

# TODO -- separate the functions into separate module
class MPGPU(Process):

    # Keep all properties that belong to an individual openpiv function within the properties dict
    # to keep the responsibilities of this class clear (multiprocessing, not keeping track of parameters)
    def __init__(self, gpuid,
                 process_num, start_index,
                 frame_list_a, frame_list_b,
                 properties):
        Process.__init__(self)
        self.gpuid = gpuid
        self.process_num = process_num
        self.start_index = start_index
        self.frame_list_a = frame_list_a
        self.frame_list_b = frame_list_b
        self.num_images = len(frame_list_a)
        self.properties = properties
        self.exceptions = 0

    def run(self):
        process_time = time()
        func = self.properties["gpu_func"]

        for i in range(self.num_images):
            frame_a, frame_b = self.frame_list_a[i], self.frame_list_b[i]    
            try:
                func(self.start_index + i, frame_a, frame_b, self.properties, gpuid=self.gpuid)
            except Exception as e:
                print "\n An exception occurred! %s" % e
                print sys.exc_info()[2].tb_lineno
                self.exceptions += 1

        print "\nProcess %d took %d seconds to finish %d image pairs (%d to %d)!" % (self.process_num,
                                                                                     time() - process_time,
                                                                                     self.num_images,
                                                                                     self.start_index,
                                                                                     self.start_index + self.num_images)
        print "\nNumber of exceptions: %d" % self.exceptions

    @staticmethod
    def load_images(image_a, image_b):
        return np.load(image_a).astype(np.int32), np.load(image_b).astype(np.int32)


def parallelize(num_items, num_processes, list_tuple, properties):
    # Properties contains data relevant to a particular openpiv function

    partitions = int(num_items/num_processes)

    print "\n Partitions Size: %d" % partitions

    process_list = []

    for i in range(num_items):
        # If we go over array bounds, stop spawning new processes
        if i*partitions > num_items:
            break
        start_index = i*partitions
        subList_A = list_tuple[0][start_index: start_index + partitions]
        subList_B = list_tuple[1][start_index: start_index + partitions]
        process = MPGPU(i % 4, i, start_index, subList_A, subList_B, properties)
        process.start()
        process_list.append(process)

    # Cleanup
    try:
        for process in process_list:
            process.join()
    except KeyboardInterrupt:
        for process in process_list:
            process.terminate()
            process.join()


# ===============================================================================
# FUNCTION DEFINITIONS
# ===============================================================================

def outlier_detection(u, v, r_thresh, mask=None, max_iter=2):
    """
    Outlier detection

    A single pair of output files is taken by this function and the function
    goes through each element of the arrays one by one. The median value of
    all the surrounding elements is taken (not including the element under
    analysis) and the median difference between the surrounding elements and
    the median value is calculated. If the difference between the element under
    analysis and the median value of the surrounding elements is greater than
    the threshold, then the element is an outlier and assigned NaN. After this
    is done for the entire array (u and v), the mask is applied to the array
    and all masked elements are assigned a value of 0. Note that only the
    outliers are assigned Nan. The arrays are then passed into
    openpiv.filters.replace_outliers which replaces all the NaN elements to be
    an average of the surrounding elements
    """

    u_out = np.copy(u).astype(float)
    v_out = np.copy(v).astype(float)

    for n in range(max_iter):
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):

                # check that the element is not a masked element (NaN)
                fin_u = np.isfinite(u_out[i, j])

                if fin_u:

                    if i == 0 and j == 0:
                        # top left
                        Ui = np.delete(u_out[:2, :2].flatten(), 0)
                    elif i == 0 and j == u.shape[1] - 1:
                        # top right
                        Ui = np.delete(u_out[:2, -2:].flatten(), 1)
                    elif i == u.shape[0] - 1 and j == 0:
                        # bottom left
                        Ui = np.delete(u_out[-2:, :2].flatten(), 2)
                    elif i == u.shape[0] - 1 and j == u.shape[1] - 1:
                        # bottom right
                        Ui = np.delete(u_out[-2:, -2:].flatten(), 3)
                    elif i == 0 and j > 0 and j < u.shape[1] - 1:
                        # top boundary
                        Ui = np.delete(u_out[:2, j - 1:j + 2].flatten(), 1)
                    elif i == u.shape[0] - 1 and j > 0 and j < u.shape[1] - 1:
                        # bottom boundary
                        Ui = np.delete(u_out[-2:, j - 1:j + 2].flatten(), 4)
                    elif i > 0 and i < u.shape[0] and j == 0:
                        # left boundary
                        Ui = np.delete(u_out[i - 1:i + 2, :2].flatten(), 2)
                    elif i > 0 and i < u.shape[0] - 1 and j == 0:
                        # right boundary
                        Ui = np.delete(u_out[i - 1:i + 2, -2:].flatten(), 3)
                    else:
                        # interior grid
                        Ui = np.delete(u_out[i - 1:i + 2, j - 1:j + 2].flatten(), 4)

                    Um = np.nanmedian(Ui)
                    rm = np.nanmedian(np.abs(Ui - Um))
                    ru0 = np.abs(u_out[i, j] - Um) / (rm + 0.1)

                    if ru0 > r_thresh:
                        u_out[i, j] = np.nan
                    if not np.isfinite(ru0):
                        u_out[i, j] = np.nan

                # check that the element is not a masked element (NaN)
                fin_v = np.isfinite(v_out[i, j])

                if fin_v:

                    if i == 0 and j == 0:
                        # top left
                        Vi = np.delete(v_out[:2, :2].flatten(), 0)
                    elif i == 0 and j == u.shape[1] - 1:
                        # top right
                        Vi = np.delete(v_out[:2, -2:].flatten(), 1)
                    elif i == u.shape[0] - 1 and j == 0:
                        # bottom left
                        Vi = np.delete(v_out[-2:, :2].flatten(), 2)
                    elif i == u.shape[0] - 1 and j == u.shape[1] - 1:
                        # bottom right
                        Vi = np.delete(v_out[-2:, -2:].flatten(), 3)
                    elif i == 0 and j > 0 and j < u.shape[1] - 1:
                        # top boundary
                        Vi = np.delete(v_out[:2, j - 1:j + 2].flatten(), 1)
                    elif i == u.shape[0] - 1 and j > 0 and j < u.shape[1] - 1:
                        # bottom boundary
                        Vi = np.delete(v_out[-2:, j - 1:j + 2].flatten(), 4)
                    elif i > 0 and i < u.shape[0] and j == 0:
                        # left boundary
                        Vi = np.delete(v_out[i - 1:i + 2, :2].flatten(), 2)
                    elif i > 0 and i < u.shape[0] - 1 and j == 0:
                        # right boundary
                        Vi = np.delete(v_out[i - 1:i + 2, -2:].flatten(), 3)
                    else:
                        Vi = np.delete(v_out[i - 1:i + 2, j - 1:j + 2].flatten(), 4)

                    Vm = np.nanmedian(Vi)
                    rm = np.nanmedian(np.abs(Vi - Vm))
                    rv0 = np.abs(v_out[i, j] - Vm) / (rm + 0.1)

                    if rv0 > r_thresh:
                        v_out[i, j] = np.nan
                    if not np.isfinite(rv0):
                        v_out[i, j] = np.nan

    # set all masked elements to zero so they are not replaced in openpiv.filters
    if mask is not None:
        u_out[mask] = 0.0
        v_out[mask] = 0.0

    print("Number of u outliers: {}".format(np.sum(np.isnan(u_out))))
    print("Percentage: {}".format(np.sum(np.isnan(u_out)) / u.size * 100))
    print("Number of v outliers: {}".format(np.sum(np.isnan(v_out))))
    print("Percentage: {}".format(np.sum(np.isnan(v_out)) / v.size * 100))

    print("Replacing Outliers")
    u_out, v_out = openpiv.filters.replace_outliers(u_out, v_out)

    return (u_out, v_out)


@measure_runtime_arg(queue=runtime_queue, function_type='total')
def replace_outliers(image_pair_num, u_file, v_file, properties, gpuid=0):
    """
    This function first loads all the output data from the output directory
    and applies the mask. All masked elements are assign NaN. A single pair of
    output files is then passed to the function "outlier_detection" where the
    outliers are identified and later replaced using openpiv.filters
    """

    # TODO: just realized **kwargs is a thing, so need to change to that after.
    output_dir = properties["out_dir"]
    mask = properties["mask"]
    r_thresh = properties["r_thresh"]

    u = np.load(u_file)
    v = np.load(v_file)

    u[mask] = np.nan
    v[mask] = np.nan

    # call outlier_detection (which replaces the outliers)
    u_out, v_out = outlier_detection(u, v, r_thresh, mask=mask)

    # save to the replacement directory
    save_files(output_dir, "u_repout_{:05d}.npy".format(image_pair_num), u_out)
    save_files(output_dir, "v_repout_{:05d}.npy".format(image_pair_num), v_out)


def interp_mask(mask, data_dir, exp=0, plot=False):
    """
    Interpolate the mask onto the output data. The mask has dimensions 996x1296
    while the the output data dimensions are much smaller (and depend on the
    minimum window size chosen)
    """

    # load the x and y location arrays from the output directory
    """
    SAVE y.npy IN THE GPU CODE SO THAT IT IS FLIPPED IN THE CORRECT ORIENTATION
    SO IT SHOULD NOT HAVE TO BE FLIPPED AGAIN HERE
    """
    x_r = np.load(data_dir + "x.npy")[0, :]
    y_r = np.load(data_dir + "y.npy")

    x_pix = np.linspace(x_r[0], x_r[-1], len(mask[0, :]))
    y_pix = np.linspace(y_r.min(), y_r.max(), len(mask[:, 0]))

    mask_int = np.empty([y_r.size, x_r.size])

    f = interp.interp2d(x_pix, y_pix, mask.astype(float))
    mask_int = f(x_r, y_r)
    mask_int = np.array(mask_int > 0.9)

    # expand the mask
    if exp > 0:
        for i in range(x_r.size):
            valid = np.where(mask_int[:, i] == 0)[0]
            if valid.size == 0:
                continue
            low = valid[0]
            high = valid[-1]
            mask_int[low - exp:low, i] = 0
            mask_int[high:high + exp + 1, i] = 0

    # plot some shit
    # if plot:
    #     mask_plot = mask.astype(float)
    #     mask_plot[mask] = np.nan
    #     mask_int_plot = mask_int.astype(float) + 10
    #     mask_int_plot[mask_int] = np.nan
    #     plt.pcolormesh(x_r, y_r, mask_int_plot, cmap="jet")
    #     plt.pcolormesh(x_pix, y_pix, mask_plot, cmap="jet")
    #     plt.colorbar()
    #     plt.show()

    return mask_int

@measure_runtime_arg(queue=runtime_queue, function_type='total')
def histogram_adjust(start_index, frame_a_file, frame_b_file, properties, gpuid=0):
    frame_a = np.load(frame_a_file).astype(np.int32)
    frame_b = np.load(frame_b_file).astype(np.int32)

    # FOR TESTING ONLY -- REMOVE ONCE WE FIGURE OUT WHAT'S ACCEPTABLE
    file = open("percentages.txt", "a+")

    pixels_a_old = frame_a.flatten()
    pixels_b_old = frame_b.flatten()
    # % Difference between the two images in pixel sum based on the first image
    p_deviation = np.sum(frame_a.flatten() - frame_b.flatten())/np.sum(pixels_a_old)

    frame_a = contrast(frame_a, r_low = 60, r_high = 90)
    frame_b = contrast(frame_b, r_low = 60, r_high = 90)

    # % Difference in pixel weight between the old and new image a
    pixels_a_new = frame_a.flatten()
    pixels_b_new = frame_b.flatten()

    a_deviation = np.sum(pixels_a_new - pixels_a_old)/np.sum(pixels_a_new)
    b_deviation = np.sum(pixels_b_new - pixels_b_old)/np.sum(pixels_b_new)

    file_string = "Image number %d: P = %f, A = %f, B = %f" % (start_index, p_deviation, a_deviation, b_deviation)
    file.write(file_string)
    file.write('\n')

    folder_prefix = properties['out_dir']

    # Save as .tif for easy error checking (manual check via image check), .npy for further calculations
    outname_A = os.path.splitext(os.path.basename(frame_a_file))[0]
    outname_B = os.path.splitext(os.path.basename(frame_b_file))[0]

    tif_path = ''.join([folder_prefix, '_tif/'])
    npy_path = ''.join([folder_prefix, '_npy/'])

    tif_file_A = ''.join([outname_A, '.tif'])
    tif_file_B = ''.join([outname_B, '.tif'])
    npy_file_A = ''.join([outname_A, '.npy'])
    npy_file_B = ''.join([outname_B, '.npy'])

    save_files(tif_path, tif_file_A, frame_a)
    save_files(tif_path, tif_file_B, frame_b)
    save_files(npy_path, npy_file_A, frame_a)
    save_files(npy_path, npy_file_B, frame_b)


@measure_runtime_arg(queue=runtime_queue, function_type='total')
def widim_gpu(start_index, frame_a_file, frame_b_file, properties, gpuid=0):

    # TODO -- Decouple these parameters from the functions below and pass them in
    # ==================================================================
    # PARAMETERS FOR OPENPIV
    # ==================================================================
    dt = properties["dt"]
    min_window_size = properties["min_window_size"]
    overlap = properties["overlap"]
    coarse_factor = properties["coarse_factor"]
    nb_iter_max = properties["nb_iter_max"]
    validation_iter = properties["validation_iter"]
    x_scale = properties["x_scale"]  # m/pixel
    y_scale = properties["y_scale"]  # m/pixel
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
    out_dir = properties["out_dir"]

    frame_a = np.load(frame_a_file).astype(np.int32)
    frame_b = np.load(frame_b_file).astype(np.int32)

    # Import after setting device number, since gpu_process has autoinit enabled.
    import openpiv.gpu_process
    x, y, u, v, mask = openpiv.gpu_process.WiDIM(frame_a, frame_b, np.ones_like(frame_a, dtype=np.int32),
                                                 min_window_size,
                                                 overlap,
                                                 coarse_factor,
                                                 dt,
                                                 validation_iter=validation_iter,
                                                 nb_iter_max=nb_iter_max)

    if x_scale != 1.0 or y_scale != 1.0:
        # scale the data
        x = x * x_scale
        u = u * x_scale
        y = y * y_scale
        v = v * y_scale

    # verify the directory exists:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # save the data
    if start_index == 0:
        np.save(out_dir + "x.npy", x)
        np.save(out_dir + "y.npy", y[::-1, 0])

    # Note: we're reversing u and v here only because the camera input is inverted. If the camera ever takes
    # images in the correct orientations, we'll have to remove u[::-1, :].
    save_files(out_dir, "u_{:05}.npy".format(start_index), u[::-1, :])
    save_files(out_dir, "v_{:05}.npy".format(start_index), v[::-1, :])


# ===================================================================================================================
# FILE READ/SAVE UTILITY & MISC
# ===================================================================================================================

@measure_runtime_arg(queue=runtime_queue, function_type='io')
def save_files(out_dir, file_name, file_list):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.save(out_dir + file_name, file_list)


def get_input_files(directory, file_name_pattern):
    # get the images (either .tif or npy), converting to npy if .tif

    file_list = sorted(glob.glob(directory + file_name_pattern))
    
    # if list is empty, files could be in tif format
    if not file_list:
        new_file_pattern = os.path.splitext(file_name_pattern)[0] + '.tif'
        file_list = sorted(glob.glob(directory + new_file_pattern))        
        
        prefix = file_name_pattern.split("*")[0]
        tif_to_npy(directory, prefix, file_list)
        file_list = sorted(glob.glob(directory + file_name_pattern))

    return file_list


def tif_to_npy(out_dir, prefix, file_list):
    for i in range(len(file_list)):
        file_read = io.imread(file_list[i])
        np.save(out_dir + prefix + "{:05}.npy".format(i), file_read)

# ===================================================================================================================
# SCRIPTING FUNCTIONS
# ===================================================================================================================


def process_images(num_images, num_processes):
    # ===============================================================================
    # DEFINE PIV & DIRECTORY VARIABLES HERE
    # ===============================================================================

    dt = 5e-6
    min_window_size = 16
    overlap = 0.50
    coarse_factor = 2
    nb_iter_max = 3
    validation_iter = 1
    x_scale = 7.45e-6  # m/pixel
    y_scale = 7.41e-6  # m/pixel

    # load the mask so that it is flipped in the correct orientation
    mask = np.load("mask.npy")[::-1, :]

    # Some threshold value for replacing spurious vectors
    r_thresh = 2.0

    # path to input and output directory
    raw_dir = "/scratch/p/psulliva/chouvinc/maria_PIV_cont/raw_data/"
    im_dir = "/scratch/p/psulliva/chouvinc/maria_PIV_cont/PIV_Cont_Output/"
    out_dir = "/scratch/p/psulliva/chouvinc/maria_PIV_cont/output_data"
    rep_dir = "/scratch/p/psulliva/chouvinc/maria_PIV_cont/replaced_data"

    # make output & replacement directory paths according to window size
    out_dir += str(min_window_size) + "/"
    rep_dir += str(min_window_size) + "/"

    camera_zero_pattern = "Camera_#0_*.npy"
    camera_one_pattern = "Camera_#1_*.npy"

    # change pattern to your filename pattern
    imA_list = get_input_files(raw_dir, camera_zero_pattern)
    imB_list = get_input_files(raw_dir, camera_one_pattern)

    # Pre-processing contrast
    contrast_properties = {"gpu_func": histogram_adjust, "out_dir": im_dir}
    parallelize(num_images, num_processes, (imA_list, imB_list), contrast_properties)

    # The images are adjusted, now refresh to include them in our lists
    im_dir += "_npy/"
    imA_list = sorted(glob.glob(im_dir + camera_zero_pattern))
    imB_list = sorted(glob.glob(im_dir + camera_one_pattern))

    # Processing images
    widim_properties = {"gpu_func": widim_gpu, "out_dir": out_dir, "dt": dt,
                        "min_window_size": min_window_size, "overlap": overlap,
                        "coarse_factor": coarse_factor, "nb_iter_max": nb_iter_max,
                        "validation_iter": validation_iter, "x_scale": x_scale,
                        "y_scale": y_scale}
    parallelize(num_images, num_processes, (imA_list, imB_list), widim_properties)

    # Post-processing
    # > Replace outliers
    # TODO: rename directories to say what data they actually contain (eg. out_dir --> vectors)
    u_list = get_input_files(out_dir, "u*.npy")
    v_list = get_input_files(out_dir, "v*.npy")

    # Interpolate the mask onto the PIV grid
    if "mask_int" not in locals():
        mask_int = interp_mask(mask, out_dir + "/", exp=2)

    routliers_properties = {
        "gpu_func": replace_outliers, "out_dir": rep_dir,
        "mask": mask_int, "r_thresh": r_thresh
        }
    parallelize(num_images, num_processes, (u_list, v_list), routliers_properties)

    aggregate_runtime_metrics(runtime_queue, num_images, num_processes, 'runtime_metrics.txt')


'''
Writes to 'runtime_metrics.txt'.
Measures the runtime by varying number of images processed
'''
def test_with_image_set_length(set_length_list):
    # Add line to queue to show which test
    image_set_length_string = '\n\n\nResults from varying image set size: \n'
    runtime_queue.put(image_set_length_string)

    for el in set_length_list:
        runtime_queue.put('\n***** Num Images: %-5f *****\n' % el)
        process_images(el, 20)


'''
Writes to 'runtime_metrics.txt'.
Measures the runtime by varying number of processes used
'''
def test_with_num_processes(number_processes_list):
    # Add line to queue to show which test
    num_processes_string = '\n\n\nResults from varying number of processes: \n'
    runtime_queue.put(num_processes_string)

    for el in number_processes_list:
        runtime_queue.put('\n***** Num Processes: %-5f *****\n' % el)
        process_images(100, el)


'''
Writes to 'runtime_metrics.txt'.
Varies runtime by window size, takes the c
'''
def test_external_set(im_dir, set_name, file_pattern, window_sizes=(64, 32, 16, 8)):
    dt = 5e-6
    overlap = 0.50
    coarse_factor = 2
    nb_iter_max = 3
    validation_iter = 1
    x_scale = 7.45e-6  # m/pixel
    y_scale = 7.41e-6  # m/pixel

    out_dir = "/scratch/p/psulliva/chouvinc/maria_PIV_cont/output_data"
    rep_dir = "/scratch/p/psulliva/chouvinc/maria_PIV_cont/replaced_data"

    imA_list = get_input_files(im_dir, file_pattern[0])
    imB_list = get_input_files(im_dir, file_pattern[1])

    if file_pattern[0] == file_pattern[1]:
        # same file pattern, means images are implictly pairs by index
        del imA_list[1::2] # delete odd entries
        del imB_list[0::2] # delete even entries

    num_images = len(imB_list)

    num_processes = 20

    for min_window_size in window_sizes:
        window_size_string  = '\n\n\nResults from %s set, with window size %d: \n' % (set_name, min_window_size)
        runtime_queue.put(window_size_string)
        # make output & replacement directory paths according to window size
        out_dir += str(min_window_size) + "/"
        rep_dir += str(min_window_size) + "/"

        # Processing images
        widim_properties = {"gpu_func": widim_gpu, "out_dir": out_dir, "dt": dt,
                            "min_window_size": min_window_size, "overlap": overlap,
                            "coarse_factor": coarse_factor, "nb_iter_max": nb_iter_max,
                            "validation_iter": validation_iter, "x_scale": x_scale,
                            "y_scale": y_scale}
        parallelize(num_images, num_processes, (imA_list, imB_list), widim_properties)

    aggregate_runtime_metrics(runtime_queue, num_images, num_processes, 'runtime_metrics.txt')


def test_multi_correlation(image_sizes, window_sizes):
    images = []

    for size in image_sizes:
        images.append(np.random.randn(size, size).astype(np.float32))

    # Use object here (going to dump into json)
    t = {}

    for i in range(len(images)):
        t_row = []
        for j in range(len(window_sizes)):
            start = time()
            for k in range(10):
                # Doing simple version of multiprocessing, since our class expects files and not np.arrays directly
                process_list = []
                p = Process(target=fake_process_images, args=(images, window_sizes, i, j, k%4))
                p.start()
                process_list.append(p)

            # Cleanup
            try:
                for process in process_list:
                    process.join()
            except KeyboardInterrupt:
                for process in process_list:
                    process.terminate()
                    process.join()
            
            print (time() - start)/50
            t_row.append((time() - start)/50)
    
        t[str(image_sizes[i])] = t_row

    with open('correlation.txt', 'a+') as file:
        json.dump(t, file)


def fake_process_images(images, window_sizes, i, j, gpuid):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

    # Imports for testing performance (after setting device number)
    from test_util.gpuFFT import fft_gpu
    from test_util.IWarrange import IWarrange_gpu
    
    for x in range(5):
        win_A, win_B = IWarrange_gpu(images[i], images[i], window_sizes[j], window_sizes[j] / 2)
        corr_gpu = fft_gpu(win_A, win_B)

def file_cleanup(file_names):
    for s in file_names:
        # Delete, regardless of extension (in case we have .csv, .xls, etc. in the future)
        for f in glob.glob(s+'.*'):
            os.remove(f)

if __name__ == "__main__":
    # Cleanup files from previous runs to keep files small (comment out to keep results after multiple runs)
    file_names = []
    file_cleanup(file_names)

    # Run tests
    #test_multi_correlation([512, 1024, 2048, 2560], [64, 32, 16, 8])
    test_with_image_set_length([20, 30, 40, 50, 60, 70, 100])
    test_with_num_processes([10, 20])

    test_external_set('/scratch/p/psulliva/cdallas/2nd_PIV_challenge_data/Case_A/images/', 'challenge', ('A*a.npy', 'A*b.npy'))
    test_external_set('/scratch/p/psulliva/cdallas/UQ_database/F001/images/', 'uncertainty', ('adj_*_data.txt', 'adj_*_data.txt'))
