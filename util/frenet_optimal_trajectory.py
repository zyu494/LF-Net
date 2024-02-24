"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
(https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame]
(https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from shapely.geometry import  Point
from util.quintic_polynomials_planner import QuinticPolynomial
from util.cubic_spline_planner import CubicSpline2D
from util.prediction import predict
from typing import Optional

SIM_LOOP = 500



show_animation = True


class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class FrenetPath:

    def __init__(self,step_time,frenet_list,extent,follow:Optional[bool]=True,keep_vel:Optional[bool]=False):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0
        self.f_flag=0
        self.k_flag=0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
        # Parameter
        self.MAX_SPEED = 15  # maximum speed [m/s]
        self.MAX_ACCEL = 10.0  # maximum acceleration [m/ss] #reset by me
        self.MAX_CURVATURE = 1.0  # maximum curvature [1/m]
        self.MAX_ROAD_WIDTH = 7.0  # maximum road width [m]
        self.D_ROAD_W = 1.0  # road width sampling length [m]
        self.DT = step_time  # time tick [s]             

        self.MAX_T = 0.5#3.0  # max prediction time [m] 
        self.MIN_T = 0.3#2.0  # min prediction time [m] 

        #special parameters in velocity keeping  
        self.TARGET_SPEED = 0  # target speed when conducting velocity keeping task
        self.D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
        self.N_S_SAMPLE = 1  # sampling number of target speed

        self.ROBOT_RADIUS = extent  # robot radius [m]

        #special parameters in car following
        self.safe_distance=2.0 #safe distance between the cars when conducting car following task
        self.follow=follow
        self.keep_vel=keep_vel
        self.merge=False
        self.m=1.0 #time distance between cars when conducting car following task
        self.delta_s=0.5 #distance deviation from the target distance

        # cost weights
        self.K_J = 0.2
        self.K_T = 0.1
        self.K_D = 1.0
        self.K_LAT = 1.0
        self.K_LON = 1.0


    def calc_frenet_paths(self,a0,frenet_list,c_speed, c_d, c_d_d, c_d_dd, s0,step_time,extent):
        frenet_paths = []
        for element in frenet_list:
            di=element[1]
            target_speed=element[-1]
            s_following=element[0]

            # generate path to each offset goal
            for di in np.arange(di-self.D_ROAD_W, di+2*self.D_ROAD_W, self.D_ROAD_W):

                # Lateral motion planning
                for Ti in np.arange(self.MIN_T, self.MAX_T, self.DT):
                    fp = FrenetPath(step_time,frenet_list,extent,follow=self.follow,keep_vel=self.keep_vel)

                    # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
                    lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

                    fp.t = [t for t in np.arange(0.0, Ti, self.DT)] #2 is revised by me
                    fp.d = [lat_qp.calc_point(t) for t in fp.t]
                    fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                    fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                    fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]
                    frenet_path_follow=[]
                    frenet_path_keep=[]
                    if self.keep_vel:
                        # Longitudinal motion planning (Velocity keeping)
                        for tv in np.arange(target_speed - self.D_T_S * self.N_S_SAMPLE,
                                            target_speed + self.D_T_S * self.N_S_SAMPLE, self.D_T_S):
                            tfp = copy.deepcopy(fp)
                            lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                            tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                            tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                            tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                            tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
                            tfp.k_flag=1

                            Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                            Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                            # square of diff from target speed
                            ds = (target_speed - tfp.s_d[1]) ** 2

                            tfp.cd = self.K_J * Jp + self.K_T * Ti + self.K_D * tfp.d[-1] ** 2
                            #tfp.cv = self.K_J * Js + self.K_T * Ti + self.K_D * ds
                            tfp.cv =  self.K_T * Ti + self.K_D * ds
                            tfp.cf = self.K_LAT * tfp.cd + self.K_LON * tfp.cv

                            frenet_path_keep.append(tfp)
                    if self.follow:
                        frenet_path_follow=self.following(s0,c_speed,a0,s_following,target_speed,fp,Ti,di)
                    frenet_paths.extend(frenet_path_keep)
                    frenet_paths.extend(frenet_path_follow)       

        return frenet_paths

    def following(self,s0,v0,a0,s_following,v_following,fp,Ti,d_following):
        a_following=3.0
        path=[]
        for s_f in np.arange(s_following-6*self.delta_s,s_following+7*self.delta_s,self.delta_s):
            v_follow=v_following+a_following*Ti
            s_follow=s_f+v_following*Ti+1/2*a_following*Ti**2
            s_target=s_follow-(self.safe_distance+self.m*v_follow)
            v_target=v_follow-self.m*a_following
            a_target=a_following
            lon_qp = QuinticPolynomial(s0, v0, 0.0, s_following, v_following, 0.0, Ti)
            tfp = copy.deepcopy(fp)
            tfp.s = [lon_qp.calc_point(t) for t in fp.t]
            tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
            tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
            tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
            tfp.f_flag=1
            Js = sum(np.power(tfp.s_ddd, 2))
            Jp = sum(np.power(tfp.d_ddd, 2))
            dv = (v_following - tfp.s_d[1]) ** 2
            ds=(s_following - tfp.s[1]) ** 2
            tfp.cd = self.K_J * Jp + self.K_T * Ti + self.K_D * tfp.d[-1] ** 2
            tfp.cv = self.K_J * Js + self.K_T * Ti + self.K_D * dv+ds
            #tfp.cv = self.K_T * Ti + self.K_D * ds
            tfp.cf = self.K_LAT * tfp.cd + self.K_LON * tfp.cv
            path.append(tfp)
        return path

        

    def calc_global_paths(self,fplist, csp):
        for fp in fplist:

            # calc global positions
            for i in range(len(fp.s)):
                ix, iy = csp.calc_position(fp.s[i])
                if ix is None:
                    break
                i_yaw = csp.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)

            # calc yaw and ds
            for j in range(len(fp.x) - 1):
                dx = fp.x[j + 1] - fp.x[j]
                dy = fp.y[j + 1] - fp.y[j]
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(math.hypot(dx, dy))
            if fp.yaw==[]:
                fplist=None
            else:
                fp.yaw.append(fp.yaw[-1])
                fp.ds.append(fp.ds[-1])

            # calc curvature
            #for i in range(len(fp.yaw) - 1):
                #fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

        return fplist


    def check_collision(self,fp, ob):
        for i in range(len(ob)):
            d = [((ix - ob[i][j][0]) ** 2 + (iy - ob[i][j][1]) ** 2)
                for j,(ix, iy) in enumerate(zip(fp.x, fp.y))]

            collision = any([di <= self.ROBOT_RADIUS ** 2 for di in d])

            if collision:
                return False

        return True

    def check_driving_area(self,fp,area):
        #for i,(x,y) in enumerate(zip(fp.x, fp.y)):
            #flag=area.contains(Point((x,y)))
        for i in range(1,2):
            pos=[fp.x[i],fp.y[i]]  
            flag=area.contains(Point(pos))
            if not flag:
                return False
      
        return True

    def check_paths(self,fplist, ob):
        ok_ind = []
        for i, _ in enumerate(fplist):
            #if any([v > self.MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
                #continue
            #elif any([abs(a) > self.MAX_ACCEL for a in
                    #fplist[i].s_dd]):  # Max accel check
                #continue
            #elif any([abs(c) > self.MAX_CURVATURE for c in
                    #fplist[i].c]):  # Max curvature check
                #continue
            prediction_time=len(fplist[i].x)*self.DT
            obs_traj=predict(obs=ob,prediction_time=prediction_time,step_time=self.DT)
            if not self.check_collision(fplist[i], obs_traj):
                continue         #delete the collision check process temporaryly
            
            #if not self.check_driving_area(fplist[i],driving_area):
                #continue
            ok_ind.append(i)

        return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob):
    FP=FrenetPath()
    fplist = FP.calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    fplist = FP.calc_global_paths(fplist, csp)
    #fplist = FP.check_paths(fplist, ob)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path


def generate_target_course(x, y):
    csp = CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


def main():
    print(__file__ + " start!!")

    # way points
    wx = [0.0, 10.0, 20.5, 35.0, 70.5]
    wy = [0.0, -6.0, 5.0, 6.5, 0.0]
    # obstacle lists
    ob = np.array([[20.0, 10.0],
                   [30.0, 6.0],
                   [30.0, 8.0],
                   [35.0, 8.0],
                   [50.0, 3.0]
                   ])

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

    # initial state
    c_speed = 10.0 / 3.6  # current speed [m/s]
    c_d = 2.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position

    area = 20.0  # animation area length [m]

    for i in range(SIM_LOOP):
        path = frenet_optimal_planning(
            csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)

        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(tx, ty)
            plt.plot(ob[:, 0], ob[:, 1], "xk")
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
            plt.grid(True)
            plt.pause(0.0001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()
