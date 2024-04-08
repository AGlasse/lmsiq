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
        im = np.zeros((32, 32))
        im[5:7, 5:7] = 1.0          # 2x2 artificial source, centred at the boundary between sub-pixel 0 and 1
        im_det = Detector.measure(im, im_pix_size)
        im_ipc_on = Ipc.apply(im, oversampling)
        im_det_ipc_on = Detector.measure(im_ipc_on, im_pix_size)
        collage = [im, im_det, im_ipc_on, im_det_ipc_on]

        png_folder = iq_filer.output_folder + '/test'
        png_folder = iq_filer.get_folder(png_folder)
        png_name = 'test_sub-pixel-illumination'
        png_path = png_folder + png_name
        title = png_name
        pane_titles = ['Zemax', 'detector', 'Zem + diffusion', 'det + diffusion']
        Plot.collage(collage, None,
                     nrowcol=(2, 2), title=title, shrink=0.25, png_path=png_path,
                     pane_titles=pane_titles,
                     aspect='equal')

        im[:, :] = 1.0
        im_det = Detector.measure(im, im_pix_size)
        im_ipc_on = Ipc.apply(im, oversampling)
        im_det_ipc_on = Detector.measure(im_ipc_on, im_pix_size)
        collage = [im, im_det, im_ipc_on, im_det_ipc_on]

        png_folder = iq_filer.output_folder + '/test'
        png_folder = iq_filer.get_folder(png_folder)
        png_name = 'test_flat-illumination'
        png_path = png_folder + png_name
        title = png_name
        pane_titles = ['Zemax', 'detector', 'Zem + diffusion', 'det + diffusion']
        Plot.collage(collage, None,
                     nrowcol=(2, 2), title=title, shrink=0.25, png_path=png_path,
                     pane_titles=pane_titles,
                     aspect='equal')
        return
