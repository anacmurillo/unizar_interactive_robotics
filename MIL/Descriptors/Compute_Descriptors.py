import Descriptors

class D_calculator():
    def __init__(self):
        self.Descriptor = Descriptors.Descriptors()

    def calculate_D(self,RGB,Depth,Style):
        All_d = {}
        for i in Style:
            descrip = None
            if i == 'ORB':
                descrip, keypoints =self.Descriptor.ComputeORB(RGB,Depth)
            elif i == 'HC':
                descrip, _ = self.Descriptor.ComputeHC_deprecated(RGB, Depth)
            elif i == 'Surf':
                descrip, keypoints = self.Descriptor.ComputeSURF(RGB, Depth)
            elif i == 'BRISK':
                descrip, keypoints = self.Descriptor.ComputeBRISK(RGB, Depth)
            elif i == 'SIFT':
                descrip, keypoints = self.Descriptor.ComputeSIFT(RGB, Depth)
            if descrip is not None:
                All_d[i] = descrip
        return All_d