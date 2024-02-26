import os
from os import listdir
import numpy as np
import pickle


class Filer:

    analysis_types = ['distortion', 'iq']
    trace_file, poly_file, wcal_file, stats_file, tf_fit_file = None, None, None, None, None
    base_results_path = None
    cube_path, iq_png_folder = None, None
    slice_results_path, dataset_results_path = None, None
    pdp_path, profiles_path, centroids_path = None, None, None

    def __init__(self, model_configuration):
        analysis_type, optical_configuration, date_stamp = model_configuration
        sub_folder = "{:s}/{:s}/{:s}".format(analysis_type, optical_configuration, date_stamp)
        self.data_folder = self.get_folder('../data/' + sub_folder)
        self.output_folder = self.get_folder('../output/' + sub_folder)
        file_leader = self.output_folder + sub_folder.replace('/', '_')
        if analysis_type == 'distortion':
            self.trace_file = file_leader + '_trace'  # All ray coordinates
            self.poly_file = file_leader + '_dist_poly.txt'
            self.wcal_file = file_leader + '_dist_wcal.txt'  # Echelle angle as function of wavelength
            self.stats_file = file_leader + '_dist_stats.txt'
            self.tf_fit_file = file_leader + '_dist_tf_fit'  # Use pkl files to write objects directly
        if analysis_type == 'iq':
            self.cube_path = Filer.get_folder(self.output_folder + '/ifu_cube')
        return

    def read_specifu_config(self):
        path = self.data_folder + 'SpecIFU_config.csv'
        with open(path, 'r') as text_file:
            text_block = text_file.read()
        line_list = text_block.split('\n')
        specifu_config = {}
        for line in line_list:
            if len(line) < 2:
                continue
            tokens = line.split(',')
            key = tokens[0].split(' ')[0]
            specifu_config[key] = []
            for token in tokens[2:]:
                val = int(token) if key in ['Conf', 'Order'] else float(token)
                specifu_config[key].append(val)
#            print(line)
        return specifu_config

    @staticmethod
    def read_pickle(pickle_path):
        file = open(pickle_path + '.pkl', 'rb')
        python_object = pickle.load(file)
        file.close()
        return python_object

    @staticmethod
    def write_pickle(pickle_path, python_object):
        file = open(pickle_path + '.pkl', 'wb')
        pickle.dump(python_object, file)
        file.close()
        return

    @staticmethod
    def get_file_list(folder, **kwargs):
        inc_tags = kwargs.get('inc_tags', [])
        exc_tags = kwargs.get('exc_tags', [])
        file_list = listdir(folder)
        for tag in inc_tags:
            file_list = [f for f in file_list if tag in f]
        for tag in exc_tags:
            file_list = [f for f in file_list if tag not in f]
        return file_list

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
    def setup_distortion_folders(data_identifier):
        optical_path, dataset, _, _, slice_locs, folder_name, config_label = data_identifier

        br_folder = '../results/' + optical_path + '/'
        Filer.base_results_path = Filer.get_folder(br_folder)
        dr_folder = br_folder + dataset
        Filer.dataset_results_path = Filer.get_folder(dr_folder)
        Filer.cube_path = Filer.get_folder(dr_folder + '/cubes')
        Filer.iq_png_folder = Filer.get_folder(dr_folder + '/png')
        return

    # @staticmethod
    # def setup_iq_folders(data_identifier):
    #     optical_path, dataset, _, _, slice_locs, folder_name, config_label = data_identifier
    #
    #     br_folder = '../results/' + optical_path + '/'
    #     Filer.base_results_path = Filer.get_folder(br_folder)
    #     dr_folder = br_folder + dataset
    #     Filer.dataset_results_path = Filer.get_folder(dr_folder)
    #     Filer.cube_path = Filer.get_folder(dr_folder + '/cubes')
    #     Filer.iq_png_folder = Filer.get_folder(dr_folder + '/png')
    #     return
    #
    def write_phase_data(self, data_table, data_type, config_no, ipc_tag):
        pickle_name = data_type + "_config{:02d}_{:s}".format(config_no, ipc_tag)
        path = self.get_folder(self.output_folder + 'phase/' + data_type) + pickle_name
        Filer.write_pickle(path, data_table)
        return path

    def read_phase_data(self, data_type, config_no, ipc_tag):
        """ Read phase data table from file
        """
        pickle_name = data_type + "_config{:02d}_{:s}".format(config_no, ipc_tag)
        path = self.get_folder(self.output_folder + 'phase/' + data_type) + pickle_name
        data_table = self.read_pickle(path)
        return data_table

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

    def write_profiles(self, data_type, data_id, xy_data, **kwargs):
        """ Write EE or LSF results_proc_detected to a csv file.
        :param data_id:
        :param xy_data:
        :param data_type:
        :return:
        """
        set_path = kwargs.get('set_path', None)
        obs_dict, ipc_tag, process_level = data_id

        # axis, x, y_perfect, y_design, y_mean, y_rms, y_mcs = xy_data
        # n_points, n_files = y_mcs.shape
        # x_max = y_mean[-1]
        lines = []

        for key in obs_dict:
            line = "{:s},{:s},".format(key, str(obs_dict[key]))
            lines.append(line)

        # Write header line for xy data
        line = ''
        for key in xy_data:
            if key != 'mc_list':
                line += "{:s},".format(key)
                continue
            mc_list = xy_data['mc_list']
            for mc_key in mc_list:
                line += "{:s},".format(mc_key)
        lines.append(line)

        n_lines, = xy_data['xdet'].shape
        for i in range(0, n_lines):
            line = ''
            for key in xy_data:
                if key != 'mc_list':
                    line += "{:13.8f},".format(xy_data[key][i])
                    continue
                mc_list = xy_data['mc_list']
                for mc_key in mc_list:
                    line += "{:13.8f},".format(mc_list[mc_key][i])
            lines.append(line)

        if set_path is None:
            folder = self.get_folder(self.output_folder + 'profiles')

            csv_name = data_type + process_level + ipc_tag
            path = folder + csv_name + '.csv'
        else:
            path = set_path
        print("Filer.write_profiles to {:s}".format(path))
        with open(path, 'w', newline='') as text_file:
            for line in lines:
                print(line, file=text_file)
        return

    def _get_results_path(self, data_id, data_type):
        dataset, slice_subfolder, ipc_tag, process_level, config_folder, mcrun_tag, axis = data_id
        folder = self.output_folder + slice_subfolder
        folder += ipc_tag + '/' + process_level + '/' + data_type
        folder = self.get_folder(folder)

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
