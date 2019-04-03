import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')
class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization=visualization
        self.colors={1:'r',-1:'g'}
        if self.visualization:
            self.fig=plt.figure()
            
data_dict={
    -1:np.array([
        [1,7],[2,8],[3,8]
    ]),
    1:np.array([
        [5,1],[6,-1],[7,3]
    ])
}