from csorca import *
import matplotlib.pyplot as plt
import random as rd

d = 25.
tau = 120.
time_step = 100.
area_size = 400.

# Configuration initiale
position1 = np.array([0., area_size/2])
destination1 = np.array([area_size, area_size/2])
heading1 = (destination1 - position1)/np.linalg.norm((position1 - destination1))

position2 = np.array([area_size, area_size/2])
destination2 = np.array([0., area_size/2])
heading2 = (destination2 - position2)/np.linalg.norm((position2 - destination2))

ac1 = Aircraft(position1, heading1, destination1)
ac2 = Aircraft(position2, heading2, destination2)

aircrafts = [ac1, ac2]

# Lancement de la situation

sim = Simulation(aircrafts, d, tau, area_size=area_size)
sim.run()