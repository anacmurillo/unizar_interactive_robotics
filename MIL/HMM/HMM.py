


class HMM:
    def __init__(self):
        self.M =[[0.9,0.1],[0.1,0.9]]
        self.classes = {'Point':0,'Show':1}
        self.last_state = None

    def new_stage(self,Class,P):
        if self.last_state==None:
            self.last_state= self.classes[Class]
            return Class,[0.5,0.5]
        else:
            new_c = self.classes[Class]
            if new_c ==0:
                p_p = P
                p_s = 1.0 - P
            else:
                p_p = 1.0 - P
                p_s = P
            if self.last_state == 0:
                Prob_P = self.M[self.last_state][0]*p_p
                Prob_S = self.M[self.last_state][1]*p_s
                self.M[self.last_state][0] = Prob_P/(Prob_P+Prob_S)
                self.M[self.last_state][1] = Prob_S/(Prob_P+Prob_S)
            else:
                Prob_P = self.M[self.last_state][0] * p_p
                Prob_S = self.M[self.last_state][1] * p_s
                self.M[self.last_state][0] = Prob_P/(Prob_P+Prob_S)
                self.M[self.last_state][1] = Prob_S/(Prob_P+Prob_S)
            # print self.M
            if Prob_P > Prob_S:
                self.last_state = 0
                return 'Point',[Prob_P,Prob_S]
            else:
                self.last_state = 1
                return 'Show',[Prob_P,Prob_S]




