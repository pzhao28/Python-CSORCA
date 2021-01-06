# coding: utf-8


""" 
    Constant Speed Optimal Reciprocal Avoidance [Durand, 2018]
    Auteur : Hatem EL KADI
    Encadrants : David Gianazza, Richard Alligier, Xavier Olive

    Octobre 2020
"""







"""
Packages imports
"""
import pygeos
import numpy as np
import time
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.affinity import translate, rotate, scale
import copy
from IPython.display import SVG, display, clear_output
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt






"""
Some useful functions
"""

# Check if a point is inside the cone from A, directed by v1 and v2 and containing B
def is_inside_the_cone(point, pointA, pointB, v1, v2):
    a_b = np.array(pointB) - np.array(pointA)
    a_point = np.array(point) - np.array(pointA)

    t1 = np.linalg.det(np.array((a_point, v1))) * np.linalg.det(np.array((a_b, v1)))
    t2 = np.linalg.det(np.array((a_point, v2))) * np.linalg.det(np.array((a_b, v2)))
    
    return t1>=0 and t2>=0


# Check if a point is inside a circle of center center and radius radius
def is_inside_the_circle(point, center, radius):
    return np.sum((center - point)**2) <= radius**2





"""
Aircraft representation
"""

class Aircraft:

    theta_max = np.radians(9)
    theta_min = np.radians(-9)
    
    def __init__(self, position, heading, destination):
        self.position = np.array(position)
        self.trajectory = [np.array(position)]
        self.heading = np.array(heading)
        self.headings = [np.array(heading)]
        self.destination = destination
        self.semi_plan = []

    @classmethod
    def set_turning_rates(theta_min = np.radians(-9), theta_max = np.radians(9)):
        Aircraft.theta_min = theta_min
        Aircraft.theta_max = theta_max


    def detect_conflict(self, other, d, tau):
        # Computing geometrical features
        a_b = other.position - self.position
        ro = np.linalg.norm(a_b)
        vr = self.heading - other.heading
        theta = np.arctan(d/ro)
        
        # tangent orientation vectors
        r1 = np.array(( (np.cos(theta), -np.sin(theta)),
                       (np.sin(theta),  np.cos(theta)) ))
        r2 = np.array(( (np.cos(theta), np.sin(theta)),
                       (-np.sin(theta),  np.cos(theta)) ))    
        a_t1 = r1.dot(a_b)
        a_t2 = r2.dot(a_b)

        # Gathering geometric features in a tuple
        geom = (vr, a_t1, a_t2, )
        # Check if relative speed lies in forbiden zone
        point = self.position + vr
        radius = (1/tau)*np.sqrt(ro**2 - d**2)
        alert = (is_inside_the_cone(point, self.position, other.position, a_t1, a_t2)
                    and (is_inside_the_circle(vr, a_b/tau, d/tau) 
                       or not is_inside_the_circle(vr, np.array((0,0)), radius)))

        return (alert, geom)


    # Compute exhaust vector (c vector in [Durand, 2018])
    def exhaust_vector(self, vr, a_t1, a_t2):
        u1 = (a_t1.dot(vr)/np.linalg.norm(a_t1)**2)*a_t1
        u2 = (a_t2.dot(vr)/np.linalg.norm(a_t2)**2)*a_t2
        c1 = u1 - vr
        c2 = u2 - vr

        return c1*(np.linalg.norm(c1) <= np.linalg.norm(c2)) + c2*(np.linalg.norm(c1) > np.linalg.norm(c2))


    def compute_semi_plan(self, other, d, tau):
        alert, geom = self.detect_conflict(other, d, tau)
        
        if alert:
            vr, a_t1, a_t2 = geom
            c = self.exhaust_vector(vr, a_t1, a_t2)
            p_ij = list(c) + list([c.dot(-(self.heading + c/2))])
            p_ji = list(-c) + list([c.dot(other.heading - c/2)])

            yield [p_ij]
            yield [p_ji]

        else:
            yield []
            yield []


    def compute_heading(self):

        if not self.semi_plan:
            #print('No conflict, I move toward destination')
            h_ideal = (self.destination - self.position)/np.linalg.norm((self.destination - self.position))
            return h_ideal
        
        else:
            #print('A conflict will occur within Tau seconds !')
            # Objective function (distance to ideal heading)
            def obj(h):
                h_star = self.destination - self.position
                h_star = (np.linalg.norm(self.heading)/np.linalg.norm(h_star))*h_star
                return (h_star[0]-h[0])**2 + (h_star[1]-h[1])**2

            def obj_jac(h):
                h_star = self.destination - self.position
                h_star = (np.linalg.norm(self.heading)/np.linalg.norm(h_star))*h_star
                return [-2*(h_star[0] - h[0]), -2*(h_star[1] - h[1])]

            def obj_hess(h):
                return np.array(( (2,0), (0,2) ))
            
            # Linear constraints (semi-plan constraints + turning rate constraints)
            r_max = np.array(( (np.cos(Aircraft.theta_max), -np.sin(Aircraft.theta_max)),
                        (np.sin(Aircraft.theta_max),  np.cos(Aircraft.theta_max)) ))
            r_min = np.array(( (np.cos(Aircraft.theta_min), -np.sin(Aircraft.theta_min)),
                        (np.sin(Aircraft.theta_min),  np.cos(Aircraft.theta_min)) ))
            h_max = r_max.dot(self.heading)
            h_min = r_min.dot(self.heading)

            turning_rate_constraints = [[h_max[1], -h_max[0]], [h_min[1], -h_min[0]]]

            lin_const_list = [sp[0:-1] for sp in self.semi_plan] + turning_rate_constraints
            nb_lin_constr = len(lin_const_list)

            lb = [-sp[-1] for sp in self.semi_plan] + list(np.zeros(len(turning_rate_constraints)))
            ub = list(np.inf*np.ones(nb_lin_constr))

            linear_constraint = LinearConstraint(lin_const_list, lb, ub)
            
            # Non linear constraint (constant speed constraint)
            def cs_constraint(h):
                return h[0]**2 + h[1]**2 - np.linalg.norm(self.heading)**2
            
            def cs_jacobian(h):
                return [2*h[0], 2*h[1]]
            
            def cs_hessian(h,v):
                return v[0]*np.array(( (2,0), (0,2) ))

            nonlinear_constraint = NonlinearConstraint(cs_constraint, 0, 0, jac=cs_jacobian, hess=cs_hessian)

            # Solving the optimization problem

            x0 = self.heading
            res = minimize(obj, x0, method='trust-constr', jac=obj_jac, hess=obj_hess,
                        constraints=[linear_constraint, nonlinear_constraint])
            return np.array(res.x)


    def reached_destination(self, epsilon=3):
        return np.linalg.norm(self.destination - self.position) <= epsilon


    def move(self, ts):
        self.position += ts*self.heading
         





"""
Simulation representation
"""

class Simulation:

    def __init__(self, aircraft, d, tau, time_step=3, area_size=500):
        self.aircraft = aircraft
        self.d = d
        self.tau = tau
        self.time_step = time_step
        self.area_size = area_size
        self.time = 0
        self.step = 0
        self.alerts = 0
        self.separation_losses = 0
        self.done = False
        self.conflict_matrix = np.zeros([len(aircraft), len(aircraft)])


    def display(self):
        # Square area
        square = Polygon([(0,0), (0, self.area_size), (self.area_size, self.area_size), (self.area_size, 0)])
        # Aircrafts
        aircrafts = [Polygon(Point(ac.position).buffer(self.d,1000), [Polygon(Point(ac.position).buffer(3,1000))]) for ac in self.aircraft]
        # Heading vectors
        arrow = Polygon([(0,-25), (37.5, 15), (25, 20), (50, 25), (45, 0), (40, 12.5)])
        arrow = scale(rotate(arrow, -45), xfact=0.80, yfact=0.80)
        arrows = [rotate(translate(arrow, xoff=ac.position[0], yoff=ac.position[1]), np.arctan2(ac.heading[1], ac.heading[0]), use_radians=True, origin=tuple(ac.position))  for ac in self.aircraft]
        # All aircraft with vectors
        m = MultiPolygon([square] + aircrafts + arrows)
        # Display
        clear_output(wait=True)
        display(SVG(m._repr_svg_()))
        #time.sleep(0.1)
    

    def draw(self):
        plt.figure()
        plt.title('{} aircraft encounter'.format(len(self.aircraft)))
        plt.xlim(0,500)
        plt.ylim(0,500)
        for i in range(len(self.aircraft)):
            plt.scatter([x[0] for x in self.aircraft[i].trajectory], [x[1] for x in self.aircraft[i].trajectory], s=7)
            for j in range(i+1, len(self.aircraft)):
                x1 = [point1[0] for point1, point2 in zip(self.aircraft[i].trajectory, self.aircraft[j].trajectory) if np.linalg.norm(point1 - point2) < self.d]
                y1 = [point1[1] for point1, point2 in zip(self.aircraft[i].trajectory, self.aircraft[j].trajectory) if np.linalg.norm(point1 - point2) < self.d]
                plt.scatter(x1, y1, s=100, c='red', alpha=0.3)

                x2 = [point2[0] for point1, point2 in zip(self.aircraft[i].trajectory, self.aircraft[j].trajectory) if np.linalg.norm(point1 - point2) < self.d]
                y2 = [point2[1] for point1, point2 in zip(self.aircraft[i].trajectory, self.aircraft[j].trajectory) if np.linalg.norm(point1 - point2) < self.d]
                plt.scatter(x2, y2, s=100, c='red', alpha=0.3)
                plt.text(x=10., y=10., s='Separation losses : {} '.format(self.separation_losses), 
                        bbox=dict(boxstyle="square",
                                  ec=tuple(np.array((1., 0.8, 0.8))*(self.separation_losses>0) + np.array((0.8, 1., 0.8))*(self.separation_losses==0)),
                                  fc=tuple(np.array((1., 0.8, 0.8))*(self.separation_losses>0) + np.array((0.8, 1., 0.8))*(self.separation_losses==0)),))


    def move(self):
        for aircraft in self.aircraft:
            if not aircraft.reached_destination(): 
                aircraft.move(self.time_step)
            aircraft.trajectory.append(copy.deepcopy(aircraft.position))
            aircraft.headings.append(copy.deepcopy(aircraft.heading))


    def run_one_step(self, display=False):
        if self.done:
            print('Simulation is over')
        else:
            N = len(self.aircraft)
            done = True
            
            for i in range(N):
                # Compute semi-plans
                for j in range(i+1, N):
                    alert = self.aircraft[i].detect_conflict(self.aircraft[j], self.d, self.tau)[0]
                    self.alerts += int(alert)
                    p_ij, p_ji = self.aircraft[i].compute_semi_plan(self.aircraft[j], self.d, self.tau)
                    self.aircraft[i].semi_plan.extend(p_ij)
                    self.aircraft[j].semi_plan.extend(p_ji)
                # Compute new heading
                new_heading = self.aircraft[i].compute_heading()
                self.aircraft[i].heading = new_heading
                # Reinitialize semi-plan to empty lists
                self.aircraft[i].semi_plan = []
                
                # update done 
                done *= self.aircraft[i].reached_destination(epsilon=3)
                self.done = done
            
            # Move aircrafts according to new headings and increment step count
            self.move()
            self.step += 1
            self.time += self.time_step

            # Check if separation losses occurs
            for i in range(N):
                for j in range(i+1, N):
                    self.separation_losses += int((np.linalg.norm(self.aircraft[i].position - self.aircraft[j].position) < self.d))
                    self.conflict_matrix[i][j] += int((np.linalg.norm(self.aircraft[i].position - self.aircraft[j].position) < self.d))
            #display
            if display:
                self.display()


    def run(self, maxiter=np.inf, display=True, mute=False, draw=True):
        start = time.time()
        # Run simulation
        while not (self.done or self.step > maxiter):
            self.run_one_step(display=display)
        # Print simulation information
        #if not mute:
            #exec_time = time.time() - start
            #mins, secs = int(exec_time//60), int(exec_time%60)
        count_conflict = np.count_nonzero(self.conflict_matrix)
            #print('Execution time : {}min{}sec \nSteps : {} \nConflict alerts : {} \nSeparation losses : {}'.format(mins, secs, self.step, self.alerts, count_conflict))
        # Plot trajectories
        if draw:
            self.draw()

        return count_conflict



"""
Running a simulation
"""

if __name__ == '__main__':

    # Initializing look-ahead time and separation distance
    TAU = 3
    SEP_DISTANCE = 1000 

    # Generating aircrafts and initial setting

    # Run a CSORCA simulation



