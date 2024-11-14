import numpy as np
from numpy import *
from numpy.linalg import lstsq
import datetime
import casadi as ca
from casadi import dot
import matplotlib.pyplot as plt
import yaml
from yaml import Loader


k1 = 4
k2 = 2
dt = 0.01


def optimal_control_casadi(x,obs,vel,R0):
    
    # Safety barrier 
    h = (x[0]-obs[0])**2 + (x[1]-obs[1])**2 - R0**2
    h_dot = 2*(x[0]-obs[0])*vel*cos(x[2]) + 2*(x[1]-obs[1])*vel*sin(x[2])

    # Lie Derivatives
    L2fh = 2*(vel**2) + k2*(h_dot + k1*h)
    LgLfh = -2*(x[0]-obs[0])*vel*sin(x[2]) + 2*(x[1]-obs[1])*vel*cos(x[2])

    opti = ca.Opti()
    u = opti.variable()

    opti.minimize(0.5*(dot(u,u)))
    opti.subject_to(LgLfh*u + L2fh >= 0)

    option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti.solver('ipopt',option)

    sol = opti.solve()
    omega = sol.value(u)

    return omega



def predict_PID(x,y,the,goal,E,old_e):
    
    kp = 1.5
    ki = 0.005
    kd = 0.25

    
    dx = goal[0] - x
    dy = goal[1] - y
    g_theta = np.arctan2(dy,dx)

    alpha = g_theta - the
    err = np.arctan2(sin(alpha),cos(alpha))

    ep = err
    ei = E + err
    ed = err - old_e

    w = kp*ep + ki*ei + kd*ed

    E += err
    old_e = err

    return w,E,old_e


def get_Lidar_points(args):

    spts = []
    center = args['obs_center']

    the = np.linspace(0,2*pi,360)

    for k in range(0,len(center)):
        cx, cy = center[k]
        r = args['obs_radius'][0]

        x = cx + r*np.cos(the)
        y = cy + r*np.sin(the)

        points = [[point[0],point[1]] for point in zip(x,y)]
        spts.extend(points)

    return spts


def find_closest_point(ground_pts,agent):

    x,y,theta = agent

    xp = np.array([x+8*np.cos(np.pi/2+theta),x+8*np.cos(-np.pi/2+theta)])
    yp = np.array([y+8*np.sin(np.pi/2+theta),y+8*np.sin(-np.pi/2+theta)])

    A = np.vstack([xp, np.ones(len(xp))]).T
    m,c = lstsq(A,yp,rcond=None)[0]

    x_obs, y_obs = zip(*ground_pts)

    bpts = []
    fpts = []
    for i in range(0,len(x_obs)):
        point = y_obs[i] - m*x_obs[i] - c
        if theta>0:
            if point < 0:
                bpts.append([x_obs[i],y_obs[i]])
            else:
                fpts.append([x_obs[i],y_obs[i]])
        elif theta<0:
            if point < 0:
                fpts.append([x_obs[i],y_obs[i]])
            else:
                bpts.append([x_obs[i],y_obs[i]])
        else:
            if x_obs[i]<xp[0] or x_obs[i]<xp[1]:
                bpts.append([x_obs[i],y_obs[i]])
            else:
                fpts.append([x_obs[i],y_obs[i]])

    if fpts == []:
        closest_point = [x+10,y+10]

    else:
        xo,yo = zip(*fpts)
        
        closest_distance = float('inf') 
        closest_point = None

        for m in range(0,len(xo)):
            distance = np.sqrt((xo[m]-x)**2 + (yo[m]-y)**2)
            if distance < closest_distance:
                closest_point = [xo[m],yo[m]]
                closest_distance = distance

    return closest_point


def plot(ax,traj,l,details,arg):

    [xg,yg], bt, _ = details
    # cpx,cpy = zip(*l[2])

    ax.scatter(0,0,c='green',lw=3,label='Start')

    lh = 0.25
    k1 = [xg - lh, xg + lh, xg + lh, xg - lh, xg - lh]
    k2 = [yg - lh, yg - lh, yg + lh, yg + lh, yg - lh]

    ax.fill(k1, k2, color='blue', label='Goal', alpha=0.3)

    centers = arg['obs_center']
    r = arg['obs_radius'][0]

    for k in range(0,len(centers)):
        cx,cy = centers[k]

        if k==0:
            circle = plt.Circle((cx,cy), r, color='dimgrey', label='Obstacles')
        else:
            circle = plt.Circle((cx,cy), r, color='dimgrey')

        ax.add_artist(circle)  

    num = []
    ind = []

    ind.append(0)
    for i in range(1,len(traj)-1):
        if traj[i,2] == 1:
            num.append(i)
        else:
            ind.append(i)
    ind.append(len(traj)-1)


    num_new = [num[0]]
    for i in range(1,len(num)-1):
        if not num[i+1] - num[i] == 1:
            num_new.append(num[i])
    num_new.append(num[-1])

    ind_new = [ind[0]]
    for i in range(1,len(ind)-1):
        if not ind[i+1] - ind[i] == 1:
            ind_new.append(ind[i])
    ind_new.append(len(traj)-1)
    


    xc = traj[ind_new[0]:ind_new[1]+1,0:1]
    yc = traj[ind_new[0]:ind_new[1]+1,1:2]
    ax.plot(xc,yc,'-',color='black',linewidth=1.0,label='PID Trajectory')

    for m in range(1,len(num_new)):

        xc = traj[num_new[m]:ind_new[m+1]+1,0:1]
        yc = traj[num_new[m]:ind_new[m+1]+1,1:2]
        ax.plot(xc,yc,'-',color='black',linewidth=1.0) 

        if m==1:
            xc = traj[ind_new[m]:num_new[m]+1,0:1]
            yc = traj[ind_new[m]:num_new[m]+1,1:2]
            ax.plot(xc,yc,'-',color='red',linewidth=2.0,label='CBF Trajectory')
        else:
            xc = traj[ind_new[m]:num_new[m]+1,0:1]
            yc = traj[ind_new[m]:num_new[m]+1,1:2]
            ax.plot(xc,yc,'-',color='red',linewidth=2.0) 

    ax.legend(loc='upper left')

    return ax

surface_pts = []

path = open('config.yaml','r')
datas = yaml.load(path,Loader)
data = datas['Obstacles']

'===========   Change 0,1,2 in experiments array as per choice   ============'
experiments = ['Goal Reaching','Local Minima','Cluttered']
experiment = experiments[0]


def main(exp,data):


    if exp=='Goal Reaching':
        '==================== 4 obstacles in a linear line, y=x ===================='
        arguments = data['Circle_linear']
        
        x,y,the = [0,0,0]
        xg,yg = [8,8]

        # CBF variables
        bt = 0.75

        lim = [x-2,xg+2]

        details = [[xg,yg],bt,lim]

    elif exp=='Local Minima':
        '==================== 7 obstacles in curve, local minima ===================='
        arguments = data['Circle_LM']
        
        x,y,the = [0,0,0]
        xg,yg = [5,5]

        # CBF variables
        bt = 0.75

        lim = [x-2,xg+2]

        details = [[xg,yg],bt,lim]

    elif exp=='Cluttered':
        '====================       8 obstacles scattered         ===================='
        arguments = data['Circle_scattered']
        
        x,y,the = [0,0,0]
        xg,yg = [20,20]

        # CBF variables
        bt = 1.0

        lim = [x-2,xg+2]

        details = [[xg,yg],bt,lim]
    
    # PID variables
    E = 0
    old_e = 0

    # Linear velocity
    vel = 0.5
    agent = np.array([x,y,the])

    goal_reached = False
    trajectory = []

    # Get synthetic Lidar points 
    surface_pts = np.array(get_Lidar_points(arguments))

    # Save the control inputs
    ci = [0]

    path_cost = 0.0
    x_prev, y_prev = 0,0

    start = datetime.datetime.now()
    while not goal_reached:
        
        cp = find_closest_point(surface_pts,agent)
        distance = np.sqrt((cp[0] - x)**2 + (cp[1] - y)**2)


        if ((distance <= bt)):
            trajectory.append([x,y,1])

            omega = optimal_control_casadi([x,y,the],cp,vel,bt)

            x += (vel*cos(the))*dt
            y += (vel*sin(the))*dt
            the += omega*dt

            the = np.arctan2(sin(the),cos(the))

        else:
            trajectory.append([x,y,0])

            omega,new_E,new_old_e = predict_PID(x,y,the,[xg,yg],E,old_e)

            E = new_E
            old_e = new_old_e

            x += (vel*cos(the))*dt
            y += (vel*sin(the))*dt
            the += omega*dt

            the = np.arctan2(sin(the),cos(the))

        agent = np.array([x,y,the])
        ci.append(omega)

        arrive_dist = np.sqrt((xg-x)**2 + (yg-y)**2)
        if arrive_dist <= 0.05:
            print("Goal reached!!!")                           
            goal_reached = True

        
        path_cost += np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
        x_prev = x
        y_prev = y

        end = datetime.datetime.now()
        delta_time = end - start
        elapsed = delta_time.total_seconds()

    return [trajectory, ci], elapsed, path_cost, details, arguments


'================   RUN THIS SNIPPET TO TEST FOR "N" ITERATIONS   ================'
if __name__ == '__main__':
    print("started!!")

    total_time = 0
    N = 1

    for _ in range(0,N):
        lists, time, pc, details, arg = main(experiment,data)

        # save the path taken by the robot
        if _ == 0:
            np.save(f'Trajectories/Unicycle_HOCBF_CBF_PID_trajectory.npy',lists[0])
            np.save(f'Plots/Unicycle_HOCBF_u_control_inputs.npy',lists[1])  
        
        total_time += time

    # Calculate average time take to solve for N runs
    avg_time = total_time/N
    print(f'Avg Time : {avg_time} \t Path cost : {pc}')


    # Plotting
    fig,ax = plt.subplots(figsize=(6,6))

    traje = np.load(f'Trajectories/Unicycle_HOCBF_CBF_PID_trajectory.npy')
    ax = plot(ax,traje,lists,details,arg)

    _,_,lim = details

    ax.set_xlim((lim[0],lim[1]))
    ax.set_ylim((lim[0],lim[1]))
    
    plt.xlabel("X-axis (m)")
    plt.ylabel("Y-axis (m)")
    plt.title(f'{experiment}')

    plt.show()




