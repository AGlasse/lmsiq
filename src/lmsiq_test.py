import numpy as np
from lms_detector import Detector
from lms_ipc import Ipc
from lmsiq_plot import Plot


class Test:

    def __init__(self):
        return

    @staticmethod
    def run(iq_filer):
        """ Test IPC/diffusion model on test images.
        """
        im_pix_size = 4.5

        oversampling = int(Detector.det_pix_size / im_pix_size)
        im_ps = np.zeros((32, 32))
        im_ps[5:7, 5:7] = 1.0          # 2x2 artificial source, centred at the boundary between sub-pixel 0 and 1
        im_ipc_ps = Ipc.apply(im_ps, oversampling)
        im_det_ps = Detector.down_sample(im_ipc_ps, im_pix_size)

        im_fl = np.zeros((32, 32))
        im_fl[:, :] = 1.0
        im_ipc_fl = Ipc.apply(im_fl, oversampling)
        im_det_fl = Detector.down_sample(im_ipc_fl, im_pix_size)
        collage = [im_ps, im_ipc_ps, im_det_ps,
                   im_fl, im_ipc_fl, im_det_fl]
        pane_titles = ['Zem point', 'point + diffusion', 'det point',
                       'Zem flat', 'flat + diffusion', 'det flat']

        png_folder = iq_filer.output_folder + '/test'
        png_folder = iq_filer.get_folder(png_folder)
        png_name = 'test_illumination'
        title = png_name
        png_path = png_folder + png_name
        Plot.images(collage,
                    nrowcol=(2, 3), title=title, pane_titles=pane_titles, aspect='equal',
                    do_log=False, vlim=[0.95, 1.05], png_path=png_path)
        png_path = png_folder + png_name + '_log'
        Plot.images(collage,
                    nrowcol=(2, 3), title=title, pane_titles=pane_titles, aspect='equal',
                    do_log=True, png_path=png_path)
        return
