import os
import numpy as np
import pickle


class Filer:

    trace_file, poly_file, wcal_file, stats_file, tf_fit_file = None, None, None, None, None
    base_results_path = None
    cube_path, png_path = None, None
    slice_results_path, dataset_results_path = None, None
    pdp_path, profiles_path, centroids_path = None, None, None
    # summary_dict = {'wave': (10, 6, '(um)'), 'order': (6, 0, '-'),
    #                 'strehl': (7, 3, '-'), 'strehl_err': (11, 3, '-'),
    #                 'srp_mc_lin': (12, 0, '-'), 'srp_mc_lin_err': (15, 0, '-'),
    #                 'srp_mc_gau': (12, 0, '-'), 'srp_mc_gau_err': (15, 0, '-'),
    #                 }

    def __init__(self, optical_configuration):
        Filer._set_file_locations(optical_configuration)
        return

    @staticmethod
    def _set_file_locations(optical_configuration):
        """ Set common file locations for use by lms_distort and lms_iq
        :return:
        """
        output_folder = "../output/distortion/{:s}".format(optical_configuration)
        output_folder = Filer.get_folder(output_folder)
        folder = output_folder + optical_configuration
        Filer.trace_file = folder + '_trace.pkl'  # All ray coordinates
        Filer.poly_file = folder + '_dist_poly.txt'
        Filer.wcal_file = folder + '_dist_wcal.txt'  # Echelle angle as function of wavelength
        Filer.stats_file = folder + '_dist_stats.txt'
        Filer.tf_fit_file = folder + '_dist_tf_fit.pkl'  # Use pkl files to write objects directly
        return

    @staticmethod
    def read_pickle(pickle_path):
        file = open(pickle_path, 'rb')
        python_object = pickle.load(file)
        file.close()
        return python_object

    @staticmethod
    def write_pickle(pickle_path, python_object):
        file = open(pickle_path, 'wb')
        pickle.dump(python_object, file)
        file.close()
        return

    @staticmethod
    def get_folder(in_path):
        tokens = in_path.split('/')
        out_path = ''
        for token in tokens:
            out_path = out_path + token + '/'
            if not os.path.exists(out_path):
                os.mkdir(out_path)
        return out_path

    @staticmethod
    def setup_folders(config):
        optical_path, dataset, _, _, slice_locs, folder_name, config_label = config

        br_folder = '../results/' + optical_path + '/'
        Filer.base_results_path = Filer.get_folder(br_folder)
        dr_folder = br_folder + dataset
        Filer.dataset_results_path = Filer.get_folder(dr_folder)
        Filer.cube_path = Filer.get_folder(dr_folder + '/cubes')
        Filer.png_path = Filer.get_folder(dr_folder + '/png')
        return

    @staticmethod
    def read_phase_data(data_type, data_id, n_runs):
        """ Read data from file
        """
        path = Filer._get_results_path(data_id, data_type)
        with open(path, 'r') as text_file:
            text_block = text_file.read()

        line_list = text_block.split('\n')
        value_list = []
        for line in line_list[2:]:
            tokens = line.split(',')
            row_values = []
            if len(tokens) > 2:
                for col in range(0, n_runs + 1):
                    row_values.append(float(tokens[col]))
                value_list.append(row_values)
        values = np.array(value_list)
        return values

    @staticmethod
    def write_centroids(data_id, det_shifts, xcen_block):

        dataset, folder_tag, config_tag, n_mcruns_tag, axis = data_id
        fmt = "{:>16s},{:>16s},{:>16s},{:>16s},{:>16s},{:>16s},"
        xcen_rec = fmt.format('dataset=', dataset, 'n_mcruns=', n_mcruns_tag, 'axis=', axis)
        xcen_rec_list = [xcen_rec]

        n_shifts, n_runs = xcen_block.shape
        # Write header line to text block
        fmt = "{:>10s},"
        xcen_rec = fmt.format('Shift',)
        for run_number in range(0, n_runs):
            fmt = "{:>12s}{:04d},"
            xcen_rec += fmt.format('Xcen_', run_number)
        xcen_rec_list.append(xcen_rec)

        for res_row, det_shift in enumerate(det_shifts):
            xcen_rec = "{:10.6f},".format(det_shift)
            for res_col in range(0, n_runs):
                xcen = xcen_block[res_row, res_col]
                xcen_rec += "{:16.6f},".format(xcen)
            xcen_rec_list.append(xcen_rec)

        path = Filer._get_results_path(data_id, 'centroids')
        with open(path, 'w', newline='') as text_file:
            for xcen_rec in xcen_rec_list:
                print(xcen_rec, file=text_file)
        return

    @staticmethod
    def read_centroids(data_id, n_runs):
        # Read data from file
        path = Filer._get_results_path(data_id, 'centroids')

        with open(path, 'r') as text_file:
            text_block = text_file.read()

        line_list = text_block.split('\n')
        centroids = []
        for line in line_list[2:]:
            tokens = line.split(',')
            cen_row = []
            if len(tokens) > 2:
                for col in range(0, n_runs + 1):
                    cen_row.append(float(tokens[col]))
                centroids.append(cen_row)
        phase_shifts = np.array(centroids)
        _, n_runs = phase_shifts.shape
        for col in range(1, n_runs):
            phase_shifts[:, col] = phase_shifts[:, col] - phase_shifts[:, 0]
        return phase_shifts

    @staticmethod
    def write_profiles(data_type, data_id, xy_data, **kwargs):
        """ Write EE or LSF results_proc_detected to a csv file.
        :param data_id:
        :param xy_data:
        :param data_type:
        :return:
        """
        set_path = kwargs.get('set_path', None)

        dataset, slice_folder, ipc_tag, process_level, config_tag, n_mcruns_tag, axis = data_id

        axis, x, y_perfect, y_design, y_mean, y_rms, y_mcs = xy_data
        n_points, n_files = y_mcs.shape
        x_max = y_mean[-1]
        rows = []

        fmt = "type=,{:s},n_points=,{:d},n_files=,{:d},dataset=,{:s},x_max=,{:16.3f}"
        run = "{:s}_{:s}".format(dataset, config_tag)
        hdr1 = fmt.format(axis, n_points, n_files, run, x_max)
        rows.append(hdr1)
        fmt = "{:>16s},{:>16s},{:>16s},{:>16s},{:>16s},"
        row = fmt.format('X/pix.', 'perfect', 'design', 'Mean', 'RMS')
        for j in range(0, n_files):
            file_id = "{:04d}".format(j)
            row += "{:>16s},".format(file_id)

        rows.append(row)

        for i in range(0, n_points):
            fmt = "{:>16.6f},{:>16.8e},{:>16.8e},{:>16.8e},{:>16.8e},"
            row = fmt.format(x[i], y_perfect[i], y_design[i], y_mean[i], y_rms[i])
            for j in range(0, n_files):
                row += "{:16.8e},".format(y_mcs[i, j])
            rows.append(row)
        if set_path is None:
            path = Filer._get_results_path(data_id, data_type)
        else:
            path = set_path
        print("Filer.write_profiles to {:s}".format(path))
        with open(path, 'w', newline='') as text_file:
            for row in rows:
                print(row, file=text_file)
        return

    @staticmethod
    def _get_results_path(data_id, data_type):
        dataset, slice_subfolder, ipc_tag, process_level, config_folder, mcrun_tag, axis = data_id
        folder = Filer.dataset_results_path + slice_subfolder
        folder += ipc_tag + '/' + process_level + '/' + data_type
        folder = Filer.get_folder(folder)

        type_tags = {'xcentroids': '_xcen', 'ycentroids': '_ycen',
                     'xfwhm_gau': '_xfwhm', 'photometry': '_phot',
                     'ee_spectral': '_eex', 'ee_spatial': '_eey',
                     'lsf_spectral': '_lsx', 'lsf_spatial': '_lsy',
                     'ee_dfp_spectral': '_eex_dfp', 'ee_dfp_spatial': '_eey_dfp',
                     'lsf_dfp_spectral': '_lsx_dfp', 'lsf_dfp_spatial': '_lsy_dfp',
                     }
        type_tag = type_tags[data_type]
        config_tag = config_folder[0:-1]
        slice_tag = slice_subfolder[:-1] + '_'
        file_name = slice_tag + ipc_tag + '_' + process_level + type_tag + '_wav_' + config_tag + '.csv'
        path = folder + file_name
        return path
