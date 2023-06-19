import numpy as np
import cv2

class Preprocessing:

    def RGB(self, bgr_img: np.array) -> np.array:
        """ converting BGR input image to RGB

        Args:
            bgr_img (np.array): input BGR image

        Returns:
            np.array: output RGB image
        """
        return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    def gray_scale(self, rgb_img: np.array) -> np.array:
        """ converting RGB input image to grayscale

        Args:
            rgb_img (np.array): input RGB image

        Returns:
            np.array: output grayscale imgae
        """
        return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    def hair_removal(self, bw_img: np.array, rgb_img:np.array) -> np.array:
        """ removing hair artifacts from RGB images 

        Args:
            bw_img (np.array): grayscaled input image
            rgb_img (np.array): RGB input image

        Returns:
            np.array: RGB image without hair artifacts
        """
        kernel = cv2.getStructuringElement(1, (10, 10))
        blackhat_img = cv2.morphologyEx(bw_img, cv2.MORPH_BLACKHAT, kernel)
        ret, threshold_img = cv2.threshold(blackhat_img, 10, 255, cv2.THRESH_BINARY)
        rm_hair_img = cv2.inpaint(rgb_img, threshold_img, 1, cv2.INPAINT_TELEA)
        return rm_hair_img

    def CLAHE(self, rm_hair_img: np.array) -> np.array:
        """ applying Contrast Limited Adaptive Histogram Equalization (CLAHE)

        Args:
            rm_hair_img (np.array): RGB image without hair artifacts

        Returns:
            np.array: RGB image without hair artifacts after applying CLAHE
        """
        lab = cv2.cvtColor(rm_hair_img, cv2.COLOR_RGB2LAB)
        lab_planes = list(cv2.split(lab))
        clahe = cv2.createCLAHE(clipLimit=1.5)
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def perform(self, bgr_image: np.array, width: float = 480, height: float = 640) -> np.array:
        """performing preprocessing on BGR image

        Args:
            bgr_image (np.array): input BGR image
            width (float, optional): output image's width. Defaults to 480.0.
            height (float, optional): output image's height. Defaults to 640.0.

        Returns:
            np.array: preprocessed RGB image
        """
        img = cv2.resize(bgr_image, (width, height))
        rgb_img = self.RGB(img)
        bw_img = self.gray_scale(rgb_img)
        hair_removed = self.hair_removal(bw_img, rgb_img)
        final = self.CLAHE(hair_removed)
        return final

