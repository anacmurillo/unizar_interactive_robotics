


class Scene:
    
    def __init__(self,RGB_front,Depth_front,Mask_front,RGB_top,Depth_top,Mask_top,files):
        self.RGB_front= RGB_front
        self.Depth_front= Depth_front
        self.Mask_front = Mask_front
        self.RGB_top = RGB_top
        self.Depth_top = Depth_top
        self.Mask_top = Mask_top
        self.files = files
        self.Rotated_RGB = None
        self.Rotated_Mask = None
        self.Rotated_Depth = None
        self.Objects = None
        self.Skeleton = None
        self.Interaction = None
        self.Values = {}

    def addObj(self,Obj):
        self.Objects = Obj

    def __str__(self):
        return self.Values.__str__()+" "+self.files.__str__()