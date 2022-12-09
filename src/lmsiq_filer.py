import numpy as np
import time
from os import listdir
from os.path import isfile, join
from astropy.io import fits

class LMSIQFiler:


    def __init__(self):
        LMSIQFiler.res_path = '../results/'
        LMSIQFiler.data_path = '../data/'
        return

    @staticmethod
    def read_file_list(rundate, folder):
        parent_folder = LMSIQFiler.data_path + 'psf_model_' + rundate + '_multi_wavelength/'     # 20201113
        path = parent_folder + folder
        file_list = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[1] == 'fits']
        return path, file_list

    @staticmethod
    def read_param_file(rundate, folder):
        parent_folder = LMSIQFiler.data_path + 'psf_model_' + rundate + '_multi_wavelength/'  # 20201113
        path = parent_folder + folder + '/' + folder + '.txt'
        pf = open(path, 'r')
        lines = pf.read().splitlines()
        pf.close()
        slice = int(lines[1].split(':')[1])
        wave = float(lines[3].split(':')[1])
        prism_angle = float(lines[4].split(':')[1])
        grating_angle = float(lines[5].split(':')[1])
        order = -1 * int(lines[6].split(':')[1])
        pix_size = float(lines[24].split(':')[1])
        return slice, wave, prism_angle, grating_angle, order, pix_size

    @staticmethod
    def read_zemax_fits(path, f):
        file_path = join(path, f)
        hdu_list = fits.open(file_path, mode='readonly')
        hdu = hdu_list[0]
        return hdu.data, hdu.header

    @staticmethod
    def write_summary(dataset, rows, id):
        summary_file_name = dataset + '_summary' + id
        path = LMSIQFiler.res_path + '/' + summary_file_name + '.csv'
        with open(path, 'w', newline='') as text_file:
            for row in rows:
                print(row, file=text_file)
        return

    @staticmethod
    def read_summary(dataset, id):
        summary_file_name = dataset + '_summary' + id
        path = LMSIQFiler.res_path + '/' + summary_file_name + '.csv'
        waves, ipcs, srps, srps_err = [], [], [], []
        fwhmspecs, fwhmspec_errs, fwhmspats, fwhmspat_errs = [], [], [], []
        strehls, strehl_errs = [], []
        with open(path, 'r') as text_file:
            records = text_file.read().splitlines()
            for record in records[2:]:
#                print(record)
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
    def write_profiles(folder, type, axis, xy_data, strehl_data, ipc_factor):
        """ Write EE or LSF results to a csv file.
        :param folder:
        :param type:
        :param axis:
        :param xs:
        :param y_mean:
        :param y_rms:
        :param y_all:
        :return:
        """
        x, y_mean, y_rms, y_all = xy_data
        n_points, n_files = y_all.shape
        x_max = y_mean[-1]
        rows = []

        strehl, strehl_err = strehl_data
        fmt = "Strehl=, {:12.6f},+-,{:12.6f},IPC factor=,{:12.3f}"
        row = fmt.format(strehl, strehl_err, ipc_factor)
        rows.append(row)

        fmt = "type,{:s},n_points,{:d},n_files,{:d},Run,{:s},x_max=,{:16.3f}"
        hdr1 = fmt.format(type, n_points, n_files, folder, x_max)
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
        res_file_name = folder + '_' + axis + '_'
        path = LMSIQFiler.res_path + '/' + res_file_name + type + '.csv'
        with open(path, 'w', newline='') as text_file:
            for row in rows:
                print(row, file=text_file)
        return

    @staticmethod
    def read_profiles(folder, type, axis):
        # Read data from file
        res_file_name = folder + '_' + axis + '_'
        path = LMSIQFiler.res_path + '/' + res_file_name + type + '.csv'
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

        xy_data = folder, axis, x, y_mean, y_rms, y_all
        strehl_data = strehl, strehl_err
        return xy_data, strehl_data, ipc_factor