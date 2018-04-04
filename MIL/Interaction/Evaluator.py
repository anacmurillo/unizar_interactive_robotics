



class Evaluator:
    def __init__(self,classes):
        self.Datos = []
        self.Classes = classes

    def add_data(self,Clas, Probability):
        C = self.Classes.index(Clas)
        self.Datos.append((C,Probability))

    def calculate_output(self):
        Output_p = [0]*len(self.Classes)
        Output_n = [0]*len(self.Classes)
        for D in self.Datos:
            Clas, P = D
            Output_n[Clas] = Output_n[Clas]+ 1
            Output_p[Clas] = Output_p[Clas]+ P
        for n in xrange(len(self.Classes)):
            if Output_n[n] == 0:
                Output_p[n] = 0
                continue
            Output_p[n] = Output_p[n]/Output_n[n]
        Result_p = Output_p.index(max(Output_p))
        Result_n = Output_n.index(max(Output_n))
        Class_p = self.Classes[Result_p]
        Class_n = self.Classes[Result_n]
        Total = sum(Output_n)
        return max(Output_n),max(Output_p),Class_p,Class_n,Total
