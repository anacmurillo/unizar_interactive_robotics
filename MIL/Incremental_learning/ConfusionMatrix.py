from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ConfusionMatrix:

    def __init__(self):
        self.names1 = ['Apple', 'Banana', 'Bowl', 'CerealBox', 'Coke', 'DietCoke','Fork', 'Glass' , 'Ketchup', 'Kleenex', 'Knife', 'Lemon', 'Lime', 'Mug', 'Noodles', 'Orange','Pringles','Plate', 'Spoon', 'BigMug', 'TeaBox', 'WaterBottle']
        self.names2 = ['Apple', 'Banana', 'Bowl', 'CerealBox', 'Coke', 'DietCoke','Fork', 'Glass' , 'Ketchup', 'Kleenex', 'Knife', 'Lemon', 'Lime', 'Mug', 'Noodles', 'Orange','Pringles', 'Plate', 'Spoon', 'BigMug', 'TeaBox', 'WaterBottle']
        self.hits = 0
        self.miss = 0
        self.total = 0
        self.sum_h = 0
        self.sum_m = 0
        self.nt = 0
        self.matrix = np.array([[0]*len(self.names1)]*(len(self.names2)))

    def add_pair(self,name1,name2):
        if self.names1 == []:
            self.names1.append(name1)
            self.names2.append(name1)
            if name1 != name2:
                self.names1.append(name2)
                self.names2.append(name2)
                self.matrix= [[0,0],[0,0]]
            else:
                self.matrix = [[0]]
        if name1 not in self.names1:
            self.names1.append(name1)
            self.names2.append(name1)
            self.matrix = [x + [0] for x in self.matrix]
            self.matrix.append([0] * len(self.names1))
        if name2 not in self.names2:
            self.names1.append(name2)
            self.names2.append(name2)
            self.matrix = [x + [0] for x in self.matrix]
            self.matrix.append([0] * len(self.names1))
        x = self.names1.index(name1)
        y = self.names2.index(name2)
        self.matrix[x][y]+=1
        self.total += 1
        if name1 == name2:
            self.hits +=1
        else:
            self.miss +=1

    def show_info(self,Name,Max_Info):
        def diagonalsum(m):
            return sum(m[i][i] for i in xrange(len(m)))
        if Max_Info:
            print("X-> Label Recogn,Y-> GT")
            print("",end="\t")
            print(self.names1)
            for i in xrange(len(self.names2)):
                if len(self.names2[i]) <4 :
                    print(self.names2[i],end="\t\t\t")
                elif len(self.names2[i]) <8:
                    print(self.names2[i],end="\t\t")
                else:
                    print(self.names2[i],end="\t")
                print(self.matrix[i])
            hits = diagonalsum(self.matrix)
            total = np.sum(self.matrix)
            print(" Total: " + total.__str__() + " Aciertos: " + hits.__str__() + " Porcentaje: " + ((hits * 100.0 / total)).__str__())
            Size = 800
            NClass = len(self.names2)
            Size_block = Size / NClass
            Matri = np.zeros((Size,Size),np.uint8)
            for i in xrange(NClass):
                for j in xrange(NClass):
                    if sum(self.matrix[i])==0:
                        Value = 0
                    else:
                        Value = self.matrix[i][j] * 1.0 / sum(self.matrix[i]) * 255
                    Matri[i * Size_block:(i + 1) * Size_block,j * Size_block:(j + 1) * Size_block] = Value
            print(Matri.shape)
            plt.imsave(Name,Matri,cmap='hot')
            plt.imshow(Matri,cmap='hot')
            plt.show()
            # print(cv2.imwrite(Name', 'Matri))

            # print(" Media Aciertos: " + (self.sum_h/self.nt).__str__() + " Media Fallos: " + (self.sum_m/self.nt).__str__())

        else:
            # hits = diagonalsum(self.matrix)
            # total = np.sum(self.matrix)
            print(" Total: "+self.total.__str__()+" Aciertos: "+self.hits.__str__()+" Porcentaje: "+ ((self.hits*100.0/self.total)).__str__())
            # print(" Total: " + self.total.__str__() + " Fallos: " + self.miss.__str__() + " Porcentaje: " + ((self.miss * 100.0 / self.total)).__str__())
            # self.sum_h += self.hits*100.0/self.total
            # self.sum_m += self.miss*100.0/self.total
            # self.nt += 1
            # self.total = 0
            # self.miss = 0
            # self.hits = 0
            return (self.hits*100.0/self.total)
