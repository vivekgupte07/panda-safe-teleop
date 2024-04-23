import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        p.connect(p.GUI)

        self.full_view = False
        self.practice = False
        if self.full_view:
            p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-20,
                                     cameraTargetPosition=[0.75, -0.25, 0.2])
        

        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
        self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))

        # k = np.random.randint(3)
        self.obstacle_positions1 = [[0.35,0.2,0.25],[0.55,-0.2,0.2],[0.63,0.1,0.1]]

        self.obstacle_positions2 = [[0.45,0.3,0.1],[0.35,-0.2,0.2],[0.65,0.1,0.2]]

        self.obstacle_positions3 = [[0.45,-0.2,0.25],[0.35,0.1,0.2],[0.65,0.1,0.3]]
                                     # Add more obstacles to this list

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        urdfRootPath=pybullet_data.getDataPath()


        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        if newPosition[2] < -0.00001:
            newPosition[2] = -0.00001
        jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[0:7]
        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])
        
        state_robot = p.getLinkState(self.pandaUid, 10)[0]
        state_camera = np.array(p.getLinkState(self.pandaUid, 11)[0])+np.array([0,-0.1,0.05])
        
        if not self.practice:
            state_obstacles = p.getBasePositionAndOrientation(self.obstacle1Uid)[0]+\
                p.getBasePositionAndOrientation(self.obstacle2Uid)[0]+\
                p.getBasePositionAndOrientation(self.obstacle3Uid)[0]
        else:
            state_obstacles=[-1,-1,-1]        
        if not self.full_view:
            p.resetDebugVisualizerCamera(cameraDistance=0.1, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=state_camera)

        p.stepSimulation()
        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 7)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        if state_object[2]<0.03 and state_object[1]<-0.05 and state_object[0]<0.4:
            done = True
        else:
            done = False
        info = {'obstacles':state_obstacles}
        self.observation = state_robot + state_fingers
        return self.observation, 0, done, info

    def reset(self, seed):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        urdfRootPath=pybullet_data.getDataPath()
        p.setGravity(0,0,-9.81)

        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

        rest_poses = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
        for i in range(7):
            p.resetJointState(self.pandaUid,i, rest_poses[i])
        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid,10, 0.08)
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
        trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.3,-0.2,-0.08], useFixedBase=True, globalScaling=0.5)
        if not self.practice:
            if seed==0:
                self.obstacle1Uid = p.loadURDF(os.path.join(urdfRootPath, "sphere_small.urdf"),basePosition=self.obstacle_positions1[0],useFixedBase=True,globalScaling=1.5)
                self.obstacle2Uid = p.loadURDF(os.path.join(urdfRootPath, "sphere_small.urdf"),basePosition=self.obstacle_positions1[1],useFixedBase=True,globalScaling=1.5)
                self.obstacle3Uid = p.loadURDF(os.path.join(urdfRootPath, "sphere_small.urdf"),basePosition=self.obstacle_positions1[2],useFixedBase=True,globalScaling=1.5)
            
            elif seed==1:
                self.obstacle1Uid = p.loadURDF(os.path.join(urdfRootPath, "sphere_small.urdf"),basePosition=self.obstacle_positions2[0],useFixedBase=True,globalScaling=1.5)
                self.obstacle2Uid = p.loadURDF(os.path.join(urdfRootPath, "sphere_small.urdf"),basePosition=self.obstacle_positions2[1],useFixedBase=True,globalScaling=1.5)
                self.obstacle3Uid = p.loadURDF(os.path.join(urdfRootPath, "sphere_small.urdf"),basePosition=self.obstacle_positions2[2],useFixedBase=True,globalScaling=1.5)
            
            elif seed==2:
                self.obstacle1Uid = p.loadURDF(os.path.join(urdfRootPath, "sphere_small.urdf"),basePosition=self.obstacle_positions3[0],useFixedBase=True,globalScaling=1.5)
                self.obstacle2Uid = p.loadURDF(os.path.join(urdfRootPath, "sphere_small.urdf"),basePosition=self.obstacle_positions3[1],useFixedBase=True,globalScaling=1.5)
                self.obstacle3Uid = p.loadURDF(os.path.join(urdfRootPath, "sphere_small.urdf"),basePosition=self.obstacle_positions3[2],useFixedBase=True,globalScaling=1.5)
            
            state_obstacles = p.getBasePositionAndOrientation(self.obstacle1Uid)[0]+\
                                p.getBasePositionAndOrientation(self.obstacle2Uid)[0]+\
                                p.getBasePositionAndOrientation(self.obstacle3Uid)[0]
        else:
            state_obstacles = [-2.6,-2.0,0.25]
            # self.obstacle1Uid = p.loadURDF(os.path.join(urdfRootPath, "sphere_small.urdf"),basePosition=self.obstacle_positions1[0],useFixedBase=True,globalScaling=1.5)

        state_object = [0.62,0.38,0.01]
        
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/003/003.urdf"), basePosition=state_object)
        state_robot = p.getLinkState(self.pandaUid, 7)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        self.observation = state_robot + state_fingers  
        info = {"obstacles":state_obstacles, "radius":0.15}

        if not self.full_view:
            p.resetDebugVisualizerCamera(cameraDistance=0.22, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=list(state_robot))

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return np.array(self.observation), info

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()
