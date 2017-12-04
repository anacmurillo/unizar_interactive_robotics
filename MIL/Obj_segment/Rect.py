class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "("+self.x.__str__()+","+self.y.__str__()+")"

class Rect(object):
    def __init__(self, p1, p2):
        '''Store the top, bottom, left and right values for points
               p1 and p2 are the (corners) in either order
        '''
        self.left   = min(p1.x, p2.x)
        self.right  = max(p1.x, p2.x)
        self.top    = min(p1.y, p2.y)
        self.bottom = max(p1.y, p2.y)

    def __str__(self):
        return "["+self.left.__str__()+","+self.right.__str__()+","+self.bottom.__str__()+","+self.top.__str__()+"]"

    def center(self):
        return (self.left+(self.right-self.left)/2,self.top+(self.bottom-self.top)/2)

    def two_point(self):
        return (self.left,self.top),(self.right,self.bottom)

    def Fixed(self):
        ratio = max(self.right-self.left,self.top-self.bottom)
        ratio = ratio/2
        center= [self.left+((self.right-self.left)/2),self.bottom+((self.top-self.bottom)/2)]
        if center[0] - ratio <= 0 or center[0] + ratio >= 640 or center[1] + ratio >= 480 or center[1]-ratio <= 0:
            s = min(480 - center[1], 640 - center[0], center[1], center[0])
            self.bottom = center[1] - s
            self.top = center[1] + s
            self.left = center[0] - s
            self.right = center[0] + s
        else:
            self.bottom = center[1] - ratio
            self.top = center[1] + ratio
            self.left = center[0] - ratio
            self.right = center[0] + ratio

def range_overlap(a_min, a_max, b_min, b_max):
    '''Neither range is completely greater than the other
    '''
    return (a_min <= b_max) and (b_min <= a_max)

def overlap(r1, r2):
    '''Overlapping rectangles overlap both horizontally & vertically
    '''
    return range_overlap(r1.left, r1.right, r2.left, r2.right) and range_overlap(r1.bottom, r1.top, r2.bottom, r2.top)

def contains(r1,p):
    return r1.left<=p[0] and r1.right>=p[0] and r1.top<=p[1] and r1.bottom >=p[1]