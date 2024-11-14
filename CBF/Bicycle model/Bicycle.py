from numpy import *
import numpy as np
from numpy.linalg import lstsq
import datetime
import casadi as ca
from casadi import dot
import matplotlib.pyplot as plt
import yaml
from yaml import Loader


k1 = 40
k2 = 2
dt = 0.01

Lr = 0.15
L = 0.25

def dynamics(states,u):

    x,y,the,vel = states
    a,w = u

    x += vel*(cos(the) - (w*sin(the)))*dt
    y += vel*(sin(the) + (w*cos(the)))*dt
    the += vel/Lr*w*dt
    vel += a

    the = np.arctan2(sin(the),cos(the))

    return x,y,the,vel


def optimal_control_casadi(x,obs,u0,R0):
    
    v = x[3]

    # Safety barrier 
    h = (x[0]-obs[0])**2 + (x[1]-obs[1])**2 - R0**2
    h_dot = 2*(x[0]-obs[0])*v*(cos(x[2])-(u0[1]*sin(x[2]))) + 2*(x[1]-obs[1])*v*(sin(x[2])+(u0[1]*cos(x[2])))

    # Lie Derivatives
    L2fh = 2*(v**2) + k2*(h_dot + k1*h)
    LgLfh = [2*(x[0]-obs[0])*cos(x[2]) + 2*(x[1]-obs[1])*sin(x[2]) , (-2*(x[0]-obs[0])*v*sin(x[2]) + 2*(x[1]-obs[1])*v*cos(x[2]))*v/Lr]

    opti = ca.Opti()
    u = opti.variable(2)

    opti.minimize(0.5*(dot(u,u)))
    opti.subject_to(dot(LgLfh,u) + L2fh >= 0)


    option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti.solver('ipopt',option)

    sol = opti.solve()
    u_opt = sol.value(u)

    return u_opt


def stanley_control_PID(x,goal,pts):
    
    ke = 1.3
    kv = 0.5

    # acceleration 
    dx = goal[0] - x[0]
    dy = goal[1] - x[1]
    # g_theta = np.arctan2(dy,dx)

    g_goal = np.sqrt(dx**2 + dy**2)
    vel_des = g_goal
    a = (vel_des - x[3])*0.05
    a = np.clip(a, -0.1, 0.1)

    # curr = np.array([x[0],x[1]])
    yaw_path = np.arctan2((pts[-1,1]-pts[0,1]),(pts[-1,0]-pts[0,0]))

    alpha = yaw_path - x[2]
    yaw_err = np.arctan2(sin(alpha),cos(alpha))

    # crosstrack error
    distances = []
    for i in range(0,len(pts)):
        d = np.sqrt((x[0]-pts[i,0])**2 + (x[1]-pts[i,1])**2)
        distances.append(d)

    ct_error = min(distances)
    yaw_ct = np.arctan2((x[1]-pts[0,1]),(x[0]-pts[0,0]))
    yaw_ct_diff = np.arctan2(sin(yaw_path - yaw_ct),cos(yaw_path - yaw_ct))

    ct_error = abs(ct_error) if yaw_ct_diff>0 else -abs(ct_error)
    yaw_ct_err = np.arctan2((ke*ct_error),(x[3]))

    # Steering angle
    delta = np.arctan2(sin(yaw_err+yaw_ct_err),cos(yaw_err+yaw_ct_err))
    delta = max(-1.22, min(1.22, delta))

    # Slip angle beta
    w = np.arctan2(np.tan(delta)*Lr,L)

    return [a,w]


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

    x,y,theta,vel = agent

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

    ax.scatter(0,0,c='green',lw=3,label='Start')

    lh = 0.25
    k1 = [xg - lh, xg + lh, xg + lh, xg - lh, xg - lh]
    k2 = [yg - lh, yg - lh, yg + lh, yg + lh, yg - lh]

    ax.fill(k1, k2, color='blue', label='Goal', alpha=0.3)

    theta = np.linspace(0,2*pi,360)

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
    ax.plot(xc,yc,'-',color='black',linewidth=4.0,label='PID Trajectory')

    for m in range(1,len(num_new)):

        xc = traj[num_new[m]:ind_new[m+1]+1,0:1]
        yc = traj[num_new[m]:ind_new[m+1]+1,1:2]
        ax.plot(xc,yc,'-',color='black',linewidth=4.0) 

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
        
        x,y,the,vel = [0,0,0,0]
        xg,yg = [8,8]

        # CBF variables
        bt = 1.25+L

        lim = [x-2,xg+2]

        details = [[xg,yg],bt,lim]

    elif exp=='Local Minima':
        '==================== 7 obstacles in curve, local minima ===================='
        arguments = data['Circle_LM']
        
        x,y,the,vel = [0,0,0,0]
        xg,yg = [5,5]

        # CBF variables
        bt = 0.75+L

        lim = [x-2,xg+2]

        details = [[xg,yg],bt,lim]

    elif exp=='Cluttered':
        '====================       8 obstacles scattered         ===================='
        arguments = data['Circle_scattered']
        
        x,y,the,vel = [0,0,0,0]
        xg,yg = [20,20]

        # CBF variables
        bt = 1.5+L

        lim = [x-2,xg+2]

        details = [[xg,yg],bt,lim]


    agent = np.array([x,y,the,vel])

    # Waypoints for PID
    xw = np.linspace(x,xg,10,endpoint=True) 
    yw = np.linspace(y,yg,10,endpoint=True)

    wp = [[p[0],p[1]] for p in zip(xw,yw)]
    wp = np.array(wp, dtype=np.float32).reshape((10,2)) 

    # Get synthetic Lidar points 
    surface_pts = np.array(get_Lidar_points(arguments))


    goal_reached = False
    trajectory = []
    ci = []
    cpt = []
    states = [[x,y,the,vel]]

    u_prev = [0,0]

    path_cost = 0.0
    x_prev, y_prev = 0,0

    start = datetime.datetime.now()
    while not goal_reached:
        
        cp = find_closest_point(surface_pts,agent)
        distance = np.sqrt((cp[0] - x)**2 + (cp[1] - y)**2)

        if ((distance <= bt)):
            trajectory.append([x,y,1])
            u_opt = optimal_control_casadi([x,y,the,vel],cp,u_prev,bt)

            x,y,the,vel = dynamics([x,y,the,vel],u_opt)
            ci.append([u_opt[0],u_opt[1],1])
            cpt.append(cp)

        else:
            trajectory.append([x,y,0])
            u_opt = stanley_control_PID([x,y,the,vel],[xg,yg],wp)

            x,y,the,vel = dynamics([x,y,the,vel],u_opt)
            ci.append([u_opt[0],u_opt[1],0])

        states.append([x,y,the,vel])
        

        agent = np.array([x,y,the,vel])
        u_prev = u_opt

        arrive_dist = np.sqrt((xg-x)**2 + (yg-y)**2)
        if arrive_dist <= 0.2:
            print("Goal reached!!!")                           
            goal_reached = True

        
        path_cost += np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
        x_prev = x
        y_prev = y

        if path_cost>=100:
            break

    end = datetime.datetime.now()
    delta_time = end - start
    elapsed = delta_time.total_seconds()

    return [trajectory, ci, states], elapsed, path_cost, details, arguments



'================   RUN THIS SNIPPET TO TEST FOR "N" ITERATIONS   ================'
if __name__ == '__main__':
    print("started!!")

    total_time = 0
    N = 1

    for _ in range(0,N):
        lists, time, pc, details, arg = main(experiment,data)

        # save the path taken by the robot
        if _ == 0:
            np.save(f'Trajectories/Bicycle_HOCBF_{experiment}_CBF_PID_trajectory.npy',lists[0])
            np.save(f'Plots/Bicycle_HOCBF_{experiment}_CBF_PID_control_inputs',lists[1])
            np.save(f'Plots/Bicycle_HOCBF_{experiment}_CBF_PID_states',lists[2])  
        
        total_time += time

    # Calculate average time take to solve for N runs
    avg_time = total_time/N
    print(f'Avg Time : {avg_time} \t Path cost : {pc}')

    fig,ax = plt.subplots(figsize=(6,6))

    traje = np.load(f'Trajectories/Bicycle_HOCBF_{experiment}_CBF_PID_trajectory.npy')

    ax = plot(ax,traje,lists,details,arg)

    _,_,lim = details

    ax.set_xlim((lim[0],lim[1]))
    ax.set_ylim((lim[0],lim[1]))
    
    plt.xlabel("X-axis (m)")
    plt.ylabel("Y-axis (m)")
    plt.title(f'{experiment}')

    plt.show()




