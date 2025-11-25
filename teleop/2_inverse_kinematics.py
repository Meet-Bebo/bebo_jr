import pinocchio
import placo
import numpy as np
import pinocchio as pin
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz
from placo_utils.tf import tf

robot = placo.RobotWrapper("../simulation/robot.urdf", placo.Flags.ignore_collisions)

viz = robot_viz(robot)

# Creating the solver
solver = placo.KinematicsSolver(robot)
solver.mask_fbase(True)

# Left arm
effector_task_left = solver.add_frame_task("left_hand", np.eye(4))
effector_task_left.configure("left_hand", "soft", 1.0, 0)

# Right arm
effector_task_right = solver.add_frame_task("right_hand", np.eye(4))
effector_task_right.configure("right_hand", "soft", 1.0, 0)

t = 0
dt = 0.01
solver.dt = dt


@schedule(interval=dt)
def loop():
    global t
    t += dt

    # Updating the target
    distance = 0.2
    effector_task_left.T_world_frame = tf.translation_matrix([distance + 0.01 * np.sin(t), 0.04 -  0.04 * np.sin(t), 0.15 + 0.05 * np.cos(t)])
    effector_task_right.T_world_frame = tf.translation_matrix([-distance - 0.01 * np.sin(t), 0.04 -  0.04 * np.sin(t), 0.15 + 0.05 * np.cos(t)])

    # Solving the IK
    solver.solve(True)
    robot.update_kinematics()

    # Displaying the robot, effector and target
    viz.display(robot.state.q)
    robot_frame_viz(robot, "left_hand")
    robot_frame_viz(robot, "right_hand")
    frame_viz("target_left", effector_task_left.T_world_frame)
    frame_viz("target_right", effector_task_right.T_world_frame)


run_loop()