import os
import numpy as np
from lms_globals import Globals
from lms_ipc import Ipc
from lms_filer import Filer


class Summariser:

    summary_dict = {'wave': (10, 6, '(um)'), 'order': (6, 0, '-'),
                    'srp_mc_lin': (12, 0, '-'), 'srp_mc_lin_err': (15, 0, '-'),
                    'srp_mc_gau': (12, 0, '-'), 'srp_mc_gau_err': (15, 0, '-'),
                    }

    def __init__(self):
        axes = Globals.axes
        for axis in axes:
            axis_tag = '_' + axis[0:4]
            ee_tag = 'ee' + axis_tag + '_'
            Summariser.summary_dict[ee_tag + 'ref'] = 12, 1, '-'
            Summariser.summary_dict[ee_tag + 'per'] = 12, 5, '-'
            Summariser.summary_dict[ee_tag + 'des'] = 12, 5, '-'
            Summariser.summary_dict[ee_tag + 'mean'] = 13, 5, '-'
            fwhm_mc_tag = 'fwhm' + axis_tag + '_'
            Summariser.summary_dict[fwhm_mc_tag + 'lin_mc'] = 17, 5, '_'
            Summariser.summary_dict[fwhm_mc_tag + 'lin_mc_err'] = 21, 5, '_'
            Summariser.summary_dict[fwhm_mc_tag + 'gau_mc'] = 17, 5, '_'
            Summariser.summary_dict[fwhm_mc_tag + 'gau_mc_err'] = 21, 5, '_'
            Summariser.summary_dict[fwhm_mc_tag + 'lin_per'] = 21, 5, '_'
            Summariser.summary_dict[fwhm_mc_tag + 'lin_des'] = 21, 5, '_'
        return

    @staticmethod
    def get_value_text(key_value_pairs):
        text = ''
        for kv_pair in key_value_pairs:
            key, value = kv_pair
            wid, dec, _ = Summariser.summary_dict[key]
            fmt = '{:>' + "{:d}".format(wid) + '.' "{:d}".format(dec) + 'f},'
            text += fmt.format(value)
        return text

    @staticmethod
    def get_hdr_text(key, **kwargs):
        unit = kwargs.get('unit', None)
        val, val_fmt = key, "{:d}"
        if unit is not None:
            val, val_fmt = unit, "{:d}"
        wid, _, unit = Summariser.summary_dict[key]
        fmt = '{:>' + val_fmt.format(wid) + 's},'
        text = fmt.format(val)
        return text

    @staticmethod
    def create_summary_header(axes):
        """ Create the header rows for the results_proc_detected summary file
        (wavelength v parameter) as a string list.
        """
        hdr1, hdr2 = '', ''
        sd = Summariser.summary_dict
        for key in sd:
            wid, dec, unit = sd[key]
            hdr1 += Summariser.get_hdr_text(key)
            hdr2 += Summariser.get_hdr_text(key, unit=unit)
        return [hdr1, hdr2]

    @staticmethod
    def write_summary(process_level, slice_subfolder, rows):
        summary_file_name = process_level + '_summary_' + Ipc.tag
        folder = Filer.get_slice_results_folder(slice_subfolder)
        path = folder + summary_file_name + '.csv'
        with open(path, 'w', newline='') as text_file:
            for row in rows:
                print(row, file=text_file)
        return

    @staticmethod
    def read_summary(process_level, slice_subfolder):
        summary_file_name = process_level + '_summary_' + Ipc.tag
        folder = Filer.get_slice_results_folder(slice_subfolder)
        path = folder + summary_file_name + '.csv'
        if not os.path.exists(path):
            print("!! File {:s} not found".format(path))
            print("   - run lmsiq with IPC = x.xxx and 'reanalyse = True'")
            return None
        profile_dict, profiles = {}, None
        with open(path, 'r') as text_file:
            records = text_file.read().splitlines()
            tokens = records[0].split(',')
            n_rows = len(records) - 2
            n_cols = len(tokens)
            for col, token in enumerate(tokens):
                profile_dict[token.lower().lstrip()] = col
            profiles = np.zeros((n_rows, n_cols))
            for row, record in enumerate(records[2:]):
                tokens = record.split(',')
                for col, token in enumerate(tokens):
                    if len(token) > 1:
                        profiles[row, col] = float(token)
        return profile_dict, profiles

    @staticmethod
    def write_phase_data(data_type, data_id, det_shifts, data_block):

        dataset, slice_folder, ipc_tag, process_level, config_tag, n_mcruns_tag, axis = data_id
        fmt = "dataset=,{:<16s},slice_folder=,{:<16s},n_mcruns=,{:<16s},axis=,{:<16s},"
        header = fmt.format(dataset, slice_folder, n_mcruns_tag, axis)
        xcen_rec_list = [header]

        n_shifts, n_runs = data_block.shape
        # Write header line to text block
        fmt = "{:>10s},"
        record = fmt.format('Shift', )
        for run_number in range(0, n_runs):
            fmt = "{:>12s}{:04d},"
            record += fmt.format('Xcen_', run_number)
        xcen_rec_list.append(record)

        for res_row, det_shift in enumerate(det_shifts):
            record = "{:10.3f},".format(det_shift)
            for res_col in range(0, n_runs):
                xcen = data_block[res_row, res_col]
                record += "{:16.6e},".format(xcen)
            xcen_rec_list.append(record)

        path = Summariser._get_results_path(data_id, data_type)
        with open(path, 'w', newline='') as text_file:
            for record in xcen_rec_list:
                print(record, file=text_file)
        return

    @staticmethod
    def read_phase_data(data_type, data_id, n_runs):
        """ Read data from file
        """
        path = Summariser._get_results_path(data_id, data_type)
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

        tag = '_cen'
        path = Summariser._get_results_path(data_id, 'centroids')
#        print("Writing centroids to {:s}".format(path))
        with open(path, 'w', newline='') as text_file:
            for xcen_rec in xcen_rec_list:
                print(xcen_rec, file=text_file)
        return

    @staticmethod
    def read_centroids(data_id, n_runs):
        # Read data from file
        tag = '_cen'
        path = Summariser._get_results_path(data_id, 'centroids')
#        print("Reading centroids from {:s}".format(path))

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
    def write_profiles(data_type, data_id, xy_data, strehl_data, ipc_factor, **kwargs):
        """ Write EE or LSF results_proc_detected to a csv file.
        :param data_id:
        :param xy_data:
        :param strehl_data:
        :param ipc_factor:
        :param data_type:
        :return:
        """
        set_path = kwargs.get('set_path', None)

        dataset, slice_folder, ipc_tag, process_level, config_tag, n_mcruns_tag, axis = data_id

        x, y_perfect, y_design, y_mean, y_rms, y_mcs = xy_data
        n_points, n_files = y_mcs.shape
        x_max = y_mean[-1]
        rows = []

        strehl, strehl_err = strehl_data
        fmt = "Strehl=, {:12.6f},+-,{:12.6f},IPC factor=,{:12.3f}"
        row = fmt.format(strehl, strehl_err, ipc_factor)
        rows.append(row)

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
            path = Summariser._get_results_path(data_id, data_type)
        else:
            path = set_path
        print("Filer.write_profiles to {:s}".format(path))
        with open(path, 'w', newline='') as text_file:
            for row in rows:
                print(row, file=text_file)
        return

    @staticmethod
    def read_profiles(data_id, profile_type):
        # Read data from file
        dataset, slice_folder, ipc_tag, process_level, config_tag, n_mcruns_tag, axis = data_id
        tag = "_{:s}_{:s}".format(profile_type, axis)
        path = Summariser._get_results_path(data_id, profile_type)
        with open(path, 'r') as text_file:
            text_block = text_file.read()
        line_list = text_block.split('\n')
        tokens = line_list[0].split(',')
        strehl = float(tokens[1])
        strehl_err = float(tokens[3])
        ipc_factor = float(tokens[5])

        hdr1 = line_list[1]
        tokens = hdr1.split(',')
        n_points = int(tokens[3])
        n_files = int(tokens[5])

        x = np.zeros(n_points)
        y_per, y_des, y_mean = np.zeros(n_points), np.zeros(n_points), np.zeros(n_points)
        y_rms = np.zeros(n_points)
        y_mcs = np.zeros((n_points, n_files))
        for i in range(0, n_points):
            line = line_list[i+3]
#            if 'lsf' in profile_type and i == 0:
#                print('Filer.readprofiles ')
#                print(path)
#                print(line)
            tokens = line.split(',')
            x[i] = float(tokens[0])
            y_per[i] = float(tokens[1])
            y_des[i] = float(tokens[2])
            y_mean[i] = float(tokens[3])
            y_rms[i] = float(tokens[4])
            for j in range(0, n_files):
                y_mcs[i, j] = float(tokens[j+5])
        xy_data = axis, x, y_per, y_des, y_mean, y_rms, y_mcs
        strehl_data = strehl, strehl_err
        return xy_data, strehl_data, ipc_factor

    @staticmethod
    def _get_results_path(data_id, data_type):
        dataset, slice_subfolder, ipc_tag, process_level, config_folder, mcrun_tag, axis = data_id

        slice_folder = Summariser._get_folder(Summariser.dataset_results_path + '/' + slice_subfolder)
        ipc_folder = Summariser._get_folder(slice_folder + Ipc.folder)
        proc_folder = Summariser._get_folder(ipc_folder + process_level + '/')
        type_folder = Summariser._get_folder(proc_folder + data_type) + '/'

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
        folder = ipc_folder + process_level + '/' + data_type
        path = folder + '/' + file_name
        if not os.path.exists(folder):
            os.mkdir(folder)
        return path
