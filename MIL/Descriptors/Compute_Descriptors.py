import Descriptors
import BoW
import Alexnet

class D_calculator():
    def __init__(self):
        self.Descriptor = Descriptors.Descriptors()
        self.BoW = BoW.BoW("")
        self.BoW.loadBoW("Descriptors/BoW_ORB.pkl")
        # self.Fc7 = Alexnet.Alexnet()

    def calculate_D(self,RGB,Depth,Style):
        All_d = {}
        for i in Style:
            descrip = None
            if i == 'ORB':
                descrip, keypoints =self.Descriptor.ComputeORB(RGB,Depth)
                if descrip is not None:
                    Obj = []
                    Obj.append((" ", descrip))
                    Hist_BoW = self.BoW.testwoBoW(Obj,1)
                    descrip = Hist_BoW[0][:]
            elif i == 'HC':
                descrip, _ = self.Descriptor.ComputeHC_deprecated(RGB, Depth)
            elif i == 'Surf':
                descrip, keypoints = self.Descriptor.ComputeSURF(RGB, Depth)
            elif i == 'BRISK':
                descrip, keypoints = self.Descriptor.ComputeBRISK(RGB, Depth)
            elif i == 'SIFT':
                descrip, keypoints = self.Descriptor.ComputeSIFT(RGB,Depth)
            # elif i == 'FC7':
                # descrip, keypoints = self.Fc7.ComputeFc7(RGB,Depth)
            if descrip is not None:
                All_d[i] = descrip
        return All_d