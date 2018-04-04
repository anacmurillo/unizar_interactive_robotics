

class Candidate():

    def __init__(self,BB_top,Position_top,BB_front,Position_front,patch,patch_T,patch_d):
        self.BB_top= BB_top
        self.patch = patch
        self.patch_T = patch_T
        self.patch_d = patch_d
        self.BB_front = BB_front
        self.Position_top = Position_top
        self.Position_front = Position_front
        if BB_top is not None and BB_front is not None:
            self.size_top = BB_top.size()
            self.size_front = BB_front.size()
        self.Label = None
        self.Values = {}
        self.Ground_Truth = None
        self.Descriptors = {}
        self.Descriptors_T = {}


    def __str__(self):
        return " Candidate : "+self.BB_front.__str__()+","+self.BB_top.__str__()

    def add_label(self, label):
        self.Label = label

    def add_descriptor(self, descriptor, style):
        self.Descriptors[style] = descriptor
