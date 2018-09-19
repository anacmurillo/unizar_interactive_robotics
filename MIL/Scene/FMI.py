

class Full_Movie():

    def __init__(self,user,action,speech,label = None):
        self.user = user
        self.action = action
        self.Label = label
        self.Images = None
        self.Objects = None
        self.Candidate_patch = None
        self.Speech = speech.split(' ')
        self.Values = {}
