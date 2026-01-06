import sys
import os
import numpy as np
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

from trajectory_generator import TrajectoryGenerator

class SimulationTracking:
    def __init__(self, box_position=None):
        self.box_position = box_position if box_position is not None else np.array([1.3, 0.07, -0.158])
        
        self.hedgehog_path = os.path.abspath(os.path.join(current_dir, '../hedgehog_data'))
        self.brt_path = os.path.abspath(os.path.join(current_dir, '../brt_data'))
        self.robot_path = os.path.abspath(os.path.join(current_dir, '../description/iiwa7_allegro_throwing.xml'))
        
        self.q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                          -2.09439510239, -3.05432619099])
        self.q_max = -self.q_min
        
        self.generator = TrajectoryGenerator(
            self.q_max, self.q_min,
            self.hedgehog_path, self.brt_path,
            self.robot_path, self.box_position
        )

    def run_multi_throwing_sim(self, mode='greedy'):
        
        box1 = np.array([0.4, 1.3, -0.1]) # blue
        box2 = np.array([1.0, 0.0, -0.1]) # red
        box3 = np.array([-0.8, 0.7, -0.1]) # yellow
        
        
        if mode == 'greedy':
            box_positions = np.array([box1, box2, box3])
            self.generator.solve_multi_targets(box_positions, animate=True, full_search=True)
        elif mode == 'naive':
            box_positions = np.array([box1, box2])
            self.generator.naive_search(box_positions, simulation=True)
        

    def run_single_throwing_sim(self, posture='posture1'):
        self.generator.solve(animate=True, posture=posture)

if __name__ == "__main__":
    sim = SimulationTracking()
    
    try:
        while True:
            # choice = input("mode: ").strip().upper()
            choice = '1'
            
            if choice == '1':
                sim.run_multi_throwing_sim(mode='greedy')
            elif choice == '2':
                sim.run_multi_throwing_sim(mode='naive')
            elif choice == '3':
                sim.run_single_throwing_sim(posture='posture1')
            elif choice == 'Q':
                break
            else:
                print("invalid choice")
                
    except KeyboardInterrupt:
        print("\n simulation tracking exit")
    except Exception as e:
        print(f"error: {e}")
