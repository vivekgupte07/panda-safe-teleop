import gym
import gym_panda
import pygame
import numpy as np
import matplotlib.pyplot as plt
import SafetyController
from plot_traj import *


id = '13' 

pygame.init()
pygame.joystick.init()

try:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Initialized Joystick: {joystick.get_name()}")

except pygame.error:
    print("Joystick not connected!")
    exit()


do_safe_control = True # on for Z
do_plot = False
do_save = False
practice = False

env = gym.make('panda-v0')



if practice:
    do_safe_control = False
    do_plot = False
    do_save = True

times = 3

for k in range(times):
    collision_instances = 0
    exp_no = str(k)
    observation, info = env.reset(seed=k)
    state_robot = np.array(observation)[:3]
    action = np.zeros(env.action_space.shape[0])
    done = False
    obstacles = np.array(info["obstacles"]).reshape(-1,3)
    radius = info["radius"]
    num_collisions = 0

    X = []
    Y = []
    Z = []
    Ux_user = []
    Uy_user = []
    Uz_user = []
    Ux = []
    Uy = []
    Uz = []
    i=0
    done = False
    while not done:
        sc = SafetyController.SafetyController(obstacles, radius, state_robot)

        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()

        ## Get joystick input
        axis_x = joystick.get_axis(0)
        axis_y = joystick.get_axis(1)
        axis_z = joystick.get_axis(4)
        gripper = joystick.get_axis(2)

        action[0] =   axis_x  #L/R
        action[1] = - axis_y #Fwd/Bwd
        action[2] = - axis_z #Up/Down

        action[3] = (1.0 - gripper) / 2.0 #Grip

        state_robot = np.array(observation)[:3]

        X.append(state_robot[0])
        Y.append(state_robot[1])
        Z.append(state_robot[2])

        u_user = action[:3]

        Ux_user.append(u_user[0])
        Uy_user.append(u_user[1])
        Uz_user.append(u_user[2] )

        if do_safe_control:    
            Lgh_x = sc.Lgh(state_robot) 
            
            psi =  sc.psi(state_robot, u_user)


            if psi < 0.0:
                uSafe = sc.u_safe(state_robot,psi)
                # uSafe[uSafe>sc.u_max] = sc.u_max
                u = u_user + uSafe
            else:
                u = u_user
        else:
            u = u_user

        Ux.append(u[0])
        Uy.append(u[1])
        Uz.append(u[2])
        
        action[:-1] = u


        observation, reward, done, _ = env.step(action)
        
        state_robot = np.array(observation)[:3]
        if sc.collision(state_robot):
            num_collisions+=1
            joystick.rumble(0, 0.7, 1000)
        else:
            if num_collisions>0:
                collision_instances +=1
            num_collisions = 0

        # env.render()

        if num_collisions >= 20:
            done = True

        if i>=20000:
            done = True
        i+=1
    if not practice:
        if do_save:
            U_Store = np.column_stack((Ux, Uy, Uz))
            U_user_Store = np.column_stack((Ux_user, Uy_user, Uz_user))
            State = np.column_stack((X,Y,Z))
            if do_safe_control:
                np.savetxt(f'data/u_safe_{str(exp_no)+"_"+id}', U_Store)
                np.savetxt(f'data/u_user_safe_{str(exp_no)+"_"+id}', U_user_Store)
                np.savetxt(f'data/X_safe_{str(exp_no)+"_"+id}', State)
                np.savetxt(f'data/collisions_safe_{str(exp_no)+"_"+id}', [collision_instances])
                np.savetxt(f'data/steps_safe_{str(exp_no)+"_"+id}', [i])

            else:
                np.savetxt(f'data/u_{str(exp_no)+"_"+id}', U_Store)
                np.savetxt(f'data/u_user_{str(exp_no)+"_"+id}', U_user_Store)
                np.savetxt(f'data/X_{str(exp_no)+"_"+id}', State)
                np.savetxt(f'data/collisions_{str(exp_no)+"_"+id}', [collision_instances])
                np.savetxt(f'data/steps_{str(exp_no)+"_"+id}', [i])



        if not do_safe_control:
            plot_3d_traj(X, Y, Z, radius, obstacles, str(exp_no)+"_"+id, do_plot, do_save)
        else:
            plot_3d_traj(X, Y, Z, radius, obstacles, 'safe_'+str(exp_no)+"_"+id, do_plot, do_save)

        if do_safe_control:
            plot_control_inputs('safe_'+str(exp_no)+"_"+id, Ux,Uy,Uz,Ux_user, Uy_user, Uz_user, do_plot, do_save)
        else:
            plot_control_inputs(str(exp_no)+"_"+id, Ux,Uy,Uz, show=do_plot, save=do_save)
env.close()
