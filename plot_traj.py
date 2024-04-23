import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_save_folder():
    save_folder = "figures/"
    return save_folder


def plot_3d_traj(x,y,z, radius, position, ID, show=True, save=False):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='.', label='Trajectory')
    ax.scatter(x[0], y[1], z[2], c='g', marker='o', s=100, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='k', marker='x', s=100, label='End')

    sphere_radius = radius
    for i in range(position.shape[0]):
        sphere_center = position[i]
        
        phi, theta = np.mgrid[0.0:2.0*np.pi:100j, 0.0:np.pi:50j]
        x_sphere = sphere_radius * np.sin(theta) * np.cos(phi) + sphere_center[0]
        y_sphere = sphere_radius * np.sin(theta) * np.sin(phi) + sphere_center[1]
        z_sphere = sphere_radius * np.cos(theta) + sphere_center[2]

        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='r', alpha=0.1, label=f'Obstacle {i}')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.legend()
    ax.axis("equal")
    if save:
        save_name = str(ID)
        save_folder = get_save_folder()
        save_path = os.path.join(save_folder, f"trajectory_{save_name}.png")
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_control_inputs(ID,u1x,u1y,u1z,u2x=None,u2y=None,u2z=None, show=True, save=False):
    time_steps = np.arange(len(u1x))

    # Create subplots
    plt.figure(figsize=(12, 6))

    # Plot for x-direction
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, u1x, label='u_x')
    if u2x is not None:
        plt.plot(time_steps, u2x, label='u_user_x', linestyle='dashed')
    plt.title('Control inputs in X-direction')
    plt.xlabel('Time step')
    plt.ylabel('Control')
    plt.legend()

    # Plot for y-direction
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, u1y, label='u_y')
    if u2y is not None:
        plt.plot(time_steps, u2y, label='u_user_y', linestyle='dashed')
    plt.title('Control Inputs in Y-direction')
    plt.xlabel('Time step')
    plt.ylabel('Control')
    plt.legend()

    # Plot for z-direction
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, u1z, label='u_z')
    if u2z is not None:
        plt.plot(time_steps, u2z, label='u_user_z', linestyle='dashed')
    plt.title('Control Inputs in Z-direction')
    plt.xlabel('Time step')
    plt.ylabel('Control')
    plt.legend()

    plt.tight_layout()

    if save:
        save_name = str(ID)
        save_folder = get_save_folder()
        save_path = os.path.join(save_folder, f"control_{save_name}.png")
        plt.savefig(save_path)

    if show:
        plt.show()

    
