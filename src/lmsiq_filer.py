import os
import numpy as np
import time


class Filer:

    res_path, profiles_path, centroids_path = None, None, None

    def __init__(self, dataset):
        res_path = '../results/' + dataset + '/'
        profiles_path = res_path + 'profiles/'
        centroids_path = res_path + 'centroids/'
        if not os.path.exists(res_path):
            os.mkdir(res_path)
        if not os.path.exists(profiles_path):
            os.mkdir(profiles_path)
        if not os.path.exists(centroids_path):
            os.mkdir(centroids_path)
        Filer.res_path = res_path
        Filer.profiles_path = profiles_path
        Filer.centroids_path = centroids_path
        return

    @staticmethod
    def create_summary_header(axes):
        """ Create the header rows for the results summary file (wavelength v parameter) as a string list.
        """
        fmt = "{:>10s},{:>8s},{:>8s},{:>10s},{:>10s},{:>12s},{:>12s},"
        hdr1 = fmt.format('Wave', 'Order', 'IPC', 'SRP', 'SRP_err', 'Strehl', 'Strehl_err')
        hdr2 = fmt.format('(um)', '-', '-', '-', '-', '-', '-')
        fmt = "{:>10s},{:>12s},{:>12s},{:>12s},{:>15s},{:>15s},"
        for axis in axes:
            ee_tag = 'EE' + axis[0:4]
            fwhm_tag = 'FWHM' + axis[0:4]
            hdr1 += fmt.format('X_' + ee_tag, ee_tag, ee_tag, ee_tag, fwhm_tag, fwhm_tag + 'err',)
            hdr2 += fmt.format('pix.', 'perfect', 'design', '<model>', 'pix.', 'pix.')
        return [hdr1, hdr2]

    @staticmethod
    def write_summary(dataset, rows, id):
        summary_file_name = dataset + '_summary' + id
        path = Filer.res_path + '/' + summary_file_name + '.csv'
        with open(path, 'w', newline='') as text_file:
            for row in rows:
                print(row, file=text_file)
        return

    @staticmethod
    def read_summary(dataset, id):
        summary_file_name = dataset + '_summary' + id
        path = Filer.res_path + '/' + summary_file_name + '.csv'
        waves, ipcs, srps, srps_err = [], [], [], []
        fwhmspecs, fwhmspec_errs, fwhmspats, fwhmspat_errs = [], [], [], []
        strehls, strehl_errs = [], []
        if not os.path.exists(path):
            print("!! File {:s} not found".format(path))
            print("   - run lmsiq with IPC = x.xxx and 'reanalyse = True'")
            return None
        with open(path, 'r') as text_file:
            records = text_file.read().splitlines()
            for record in records[2:]:
                tokens = record.split(',')
                waves.append(float(tokens[0]))
                ipcs.append(float(tokens[2]))
                srps.append(float(tokens[3]))
                srps_err.append(float(tokens[4]))
                strehls.append(float(tokens[5]))
                strehl_errs.append(float(tokens[6]))
                fwhmspecs.append(float(tokens[11]))
                fwhmspec_errs.append(float(tokens[12]))
                fwhmspats.append(float(tokens[17]))
                fwhmspat_errs.append(float(tokens[18]))
        p_waves, p_ipcs = np.array(waves), np.array(ipcs)
        p_srps = np.array(srps), np.array(srps_err)
        p_strehls = np.array(strehls), np.array(strehl_errs)
        p_fwhmspecs = np.array(fwhmspecs), np.array(fwhmspec_errs)
        p_fwhmspats = np.array(fwhmspats), np.array(fwhmspat_errs)
        profile = id, p_waves, p_ipcs, p_srps, p_strehls, p_fwhmspecs, p_fwhmspats
        return profile

    @staticmethod
    def write_centroids(data_id, det_shifts, xcen_block, fwhm_block):

        dataset, config_tag, axis = data_id
        fmt = "{:>16s},{:>16s},{:>16s},{:>16s},{:>16s},{:>16s},"
        rec = fmt.format('dataset=', dataset, 'config=', config_tag, 'axis=', axis)
        rec_list = [rec]

        n_shifts, n_runs = xcen_block.shape
        # Write header line to text block
        fmt = "{:>10s},"
        rec = fmt.format('Shift', )
        for run_number in range(0, n_runs):
            fmt = "{:>10s}_{:04d},{:>10s}_{:04d},"
            rec += fmt.format('Xcen', run_number, 'FWHM', run_number)
        rec_list.append(rec)

        for res_row, det_shift in enumerate(det_shifts):
            rec = "{:10.6f},".format(det_shift)
            for res_col in range(0, n_runs):
                xcen = xcen_block[res_row, res_col]
                fwhm = fwhm_block[res_row, res_col]
                rec += "{:16.6f},{:16.6f},".format(xcen, fwhm)
            rec_list.append(rec)

        res_file_name = dataset + '_' + config_tag + '_' + axis + '_centroids'
        path = Filer.centroids_path + res_file_name + '.csv'
        with open(path, 'w', newline='') as text_file:
            for rec in rec_list:
                print(rec, file=text_file)
        return

    @staticmethod
    def read_centroids(data_id):
        # Read data from file
        dataset, config_tag, axis = data_id
        res_file_name = dataset + '_' + config_tag + '_' + axis + '_centroids'
        path = Filer.centroids_path + res_file_name + '.csv'
        with open(path, 'r') as text_file:
            text_block = text_file.read()

        line_list = text_block.split('\n')
        centroids = []
        for line in line_list[2:]:
            tokens = line.split(',')
            if len(tokens) > 2:
                det_shift, xcen, fwhm = float(tokens[0]), float(tokens[1]), float(tokens[2])
                phase_error = xcen - det_shift
                centroid = det_shift, xcen, fwhm, phase_error
                centroids.append(centroid)
        return np.array(centroids)

    @staticmethod
    def write_profiles(data_id, xy_data, strehl_data, ipc_factor, profile_type):
        """ Write EE or LSF results to a csv file.
        :param data_id:
        :param xy_data:
        :param strehl_data:
        :param ipc_factor:
        :return:
        """
        dataset, tag, axis = data_id
        x, y_mean, y_rms, y_all = xy_data
        n_points, n_files = y_all.shape
        x_max = y_mean[-1]
        rows = []

        strehl, strehl_err = strehl_data
        fmt = "Strehl=, {:12.6f},+-,{:12.6f},IPC factor=,{:12.3f}"
        row = fmt.format(strehl, strehl_err, ipc_factor)
        rows.append(row)

        fmt = "type,{:s},n_points,{:d},n_files,{:d},Run,{:s},x_max=,{:16.3f}"
        run = "{:s}_{:s}".format(dataset, tag)
        hdr1 = fmt.format(axis, n_points, n_files, run, x_max)
        rows.append(hdr1)
        fmt = "{:>16s},{:>16s},{:>16s},"
        row = fmt.format('X/pix.','Mean','RMS')
        for j in range(0, n_files):
            file_id = "{:04d}".format(j-2)
            if j == 0:
                file_id = 'perfect'
            if j == 1:
                file_id = 'design'
            row += "{:>16s},".format(file_id)
        rows.append(row)

        for i in range(0, n_points):
            row = "{:>16.6f},{:>16.8e},{:>16.8e},".format(x[i], y_mean[i], y_rms[i])
            for j in range(0, n_files):
                row += "{:16.8e},".format(y_all[i, j])
            rows.append(row)
        gmt = time.gmtime()
        fmt = '{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}'
        timestamp = fmt.format(gmt[0], gmt[1], gmt[2], gmt[3], gmt[4], gmt[5])
        res_file_name = dataset + '_' + tag + '_' + axis + '_' + profile_type
        path = Filer.profiles_path + res_file_name + '.csv'
        with open(path, 'w', newline='') as text_file:
            for row in rows:
                print(row, file=text_file)
        return

    @staticmethod
    def read_profiles(data_id, profile_type):
        # Read data from file
        dataset, tag, axis = data_id
        res_file_name = dataset + '_' + tag + '_' + axis + '_' + profile_type
        path = Filer.profiles_path + res_file_name + '.csv'
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
        y_mean = np.zeros(n_points)
        y_rms = np.zeros(n_points)
        y_all = np.zeros((n_points, n_files))
        for i in range(0, n_points):
            line = line_list[i+3]
            tokens = line.split(',')
            x[i] = float(tokens[0])
            y_mean[i] = float(tokens[1])
            y_rms[i] = float(tokens[2])
            for j in range(0, n_files):
                y_all[i,j] = float(tokens[j+3])

        xy_data = dataset, axis, x, y_mean, y_rms, y_all
        strehl_data = strehl, strehl_err
        return xy_data, strehl_data, ipc_factor