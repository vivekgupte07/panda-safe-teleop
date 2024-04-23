import numpy as np
import matplotlib.pyplot as plt


class SafetyController:
    def __init__(self, obs, radius,state):
        self.A = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]])
        
        self.B = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]])
        if np.ndim(obs) > 1:
            self.get_closest_obstacle(obs, state)
        else:
            self.x_obs = obs
        self.r_obs = radius
        self.gamma =  0.1



        self.u_max = 1.0

    def get_closest_obstacle(self, obs_list, X):
        dist = np.zeros((obs_list.shape[0]))
        for i in range(obs_list.shape[0]):
            dist[i] = np.sqrt((obs_list[i][0]-X[0])**2+(obs_list[i][1]-X[1])**2+(obs_list[i][2]-X[2])**2)
        idx = np.argmin(dist)
        self.x_obs = obs_list[idx,:]



    def plot_circle(center, radius):
        theta = np.linspace(0, 2*np.pi, 100)
        circle = np.array([center[0] + radius * np.cos(theta),
                        center[1] + radius * np.sin(theta)])
        plt.plot(*circle, label='Circle')
        plt.scatter(*center, color='red', label='Center')

    def h(self, x):
        h_x = (x[0]-self.x_obs[0])**2 + (x[1]-self.x_obs[1])**2 + (x[2]-self.x_obs[2])**2 - self.r_obs**2
        return h_x

    def grad_h(self, x):
        grad_h_x = 2*(x-self.x_obs)
        return grad_h_x

    def x_dot(self, u):
        return self.B@u

    def Lgh(self, x):
        return self.grad_h(x).T@self.B  

    def psi(self, x, u):
        return self.Lgh(x)@u + self.gamma * self.h(x)

    def u_safe(self, x, psi_xu):
        return -psi_xu*self.Lgh(x)/np.dot(self.Lgh(x),self.Lgh(x))

    def collision(self, x):
        dist = np.sqrt((x[0]-self.x_obs[0])**2 + (x[1]-self.x_obs[1])**2 + (x[2]-self.x_obs[2])**2)
        # print(dist)
        if dist <= self.r_obs:
            return True
        else:
            return False