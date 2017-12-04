

class Candidate():

    def __init__(self,BB_top,Position_top,BB_front,Position_front,patch):
        self.BB_top= BB_top
        self.patch = patch
        self.BB_front = BB_front
        self.Position_top = Position_top
        self.Position_front = Position_front
        self.Label = None
        self.Ground_Truth = None
        self.Descriptors = {}

    def __str__(self):
        return " Candidate : "+self.BB_front.__str__()+","+self.BB_top.__str__()

    def add_label(self, label):
        self.Label = label

    def add_descriptor(self, descriptor, style):
        self.Descriptors[style] = descriptor
