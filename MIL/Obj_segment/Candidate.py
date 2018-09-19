import math

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

    def get_center(self,trans,rot):
        def do_point_transform(data,trans,rot):
            [x, y,z] = data
            print(x, y)
            center_point = upper_left_to_zero_center(x, y, 640, 480)
            position = calc_geometric_location(center_point[0], center_point[1], z, 640, 480,trans,rot)
            print position

        def upper_left_to_zero_center(x, y, width, height):
            '''Change referential from center to upper left'''
            return (x - int(width / 2), y - int(height / 2))

        def calc_geometric_location(x_pixel, y_pixel, kinect_z, width, height,trans,rot):
            f = width / (2 * math.tan(math.radians(57 / 2)))  # 57 = fov angle in kinect spects
            d = kinect_z / f
            x = d * x_pixel
            y = d * y_pixel

            [transz, transx, transy] = trans
            point = [kinect_z + transz, x + transx, y + transy]
            return point_rotation_by_quaternion(point, rot)

            # https://math.stackexchange.com/questions/40164/how-do-you-rotate-a-vector-by-a-unit-quaternion

        def quaternion_mult(q, r):
            '''Quaternion multiplication'''
            return [r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
                    r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
                    r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
                    r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]]

        def point_rotation_by_quaternion(point, q):
            '''Point rotation by a quaternion'''
            [x, y, z] = point
            r = [0, x, -1 * y, -1 * z]
            q_conj = [q[0], 1 * q[1], 1 * q[2], 1 * q[3]]

            return quaternion_mult(quaternion_mult(q, r), q_conj)[1:]

        x,y = self.BB_top.center()
        cx,cy = self.BB_top.cent()
        dep = self.patch_d[cx-10:cx+10,cy-10:cy+10]
        z = dep[dep != 0].mean()
        return do_point_transform((x,y,z),trans,rot)