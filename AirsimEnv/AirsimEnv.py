import gym
import math
import numpy as np
from airsim import CarClient, CarControls, ImageRequest, ImageType, Pose, Vector3r


# (Credit to Alessandro Riboni for this script)
class AirsimEnv(gym.Env):
    """Custom environment for AirSim simulator"""

   # setting connection with AirSim simulator
    def __init__(self, ip, port):
        super(AirsimEnv, self).__init__()
        self.client = CarClient(ip=ip, port=port)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.reward_points = self._init_reward_points()

    # loading points to compute reward
    def _init_reward_points(self):
        road_points = []
        with open('/home/vz21081/work/cs/csp-drive-rl/AirsimEnv/final_reward_points.csv', newline="") as csvfile:
            for point_values in csvfile:
                point_values = point_values.split(",")
                first_point = np.array([float(point_values[0]), float(point_values[1]), 0])
                second_point = np.array([float(point_values[2]), float(point_values[3]), 0])
                road_points.append(tuple((first_point, second_point)))

        return road_points

    # to erase animals in the environment
    def _animals_out(self, neighbourhood=True):

        if neighbourhood:
            animals = ["RaccoonBP2_85", "RaccoonBP3_154", "RaccoonBP4_187", "RaccoonBP_50", "DeerBothBP2_19",
                       "DeerBothBP3_43", "DeerBothBP4_108", "DeerBothBP5_223", "DeerBothBP_12"]
        else:
            animals = []
            objects = self.client.simListSceneObjects()
            for obj in objects:
                if obj[0:7] == "Raccoon" or obj[0:8] == "DeerBoth":
                    animals.append(obj)

        for animal in animals:
            pose = self.client.simGetObjectPose(animal)
            pose.position.x_val += 500
            pose.position.y_val += 500
            self.client.simSetObjectPose(animal, pose)

    # restores the environment at each start of the episode, specifying the starting points as a parameter of the function
    # output: observation (image of the vehicle's front camera)
    def reset(self, init_point):
        """ Reset environment from the initial point init_point"""
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self._animals_out()

        point_pose = Pose()
        point_pose.position = Vector3r(init_point[0], init_point[1], init_point[2])
        point_pose.orientation.w_val = init_point[3]
        point_pose.orientation.x_val = init_point[4]
        point_pose.orientation.y_val = init_point[5]
        point_pose.orientation.z_val = init_point[6]
        car_controls = CarControls()
        car_controls.brake = 0
        self.client.setCarControls(car_controls)
        self.client.simSetVehiclePose(point_pose, True)

        observation = self.observe()
        return observation

    # retrieves, converts and returns the image of the vehicle's front camera
    def observe(self):

        response = self.client.simGetImages([ImageRequest(0, ImageType.Scene, False, False)])[0]
        size_response = (response.height, response.width, 3)
        if size_response == (144, 256, 3):
            img1d_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img3d_rgb = img1d_rgb.reshape(response.height, response.width, 3)
        else:
            print("Something bad happened. It is returned a black frame.")
            img3d_rgb = np.ones((144, 256, 3)).astype(np.uint8)

        return img3d_rgb

    # choose action (based on agent's output)
    def _perform_action(self, action, car_state):

        car_controls = CarControls()
        action_space = [0, 0.5, -0.5, 1, -1]
        car_controls.steering = action_space[action]

        if car_state.speed < 10:
            car_controls.throttle = 1
            car_controls.brake = 0
        else:
            car_controls.throttle = 0
            car_controls.brake = 1

        self.client.setCarControls(car_controls)

    # it is similar to classic step function of RL algorithms
    # perform action under AirSim, check status of the vehicle, compute the reward, check if the episode is ended
    # and receive new observation
    def step(self, action):

        car_state = self.client.getCarState()
        self._perform_action(action, car_state)
        info_collision = self.client.simGetCollisionInfo()
        car_state = self.client.getCarState()
        reward, out_street = self.compute_reward(car_state, info_collision)

        # check if the episode is done
        done = self.is_done(car_state, info_collision, out_street)

        # log info
        info = {}

        # get observation
        observation = self.observe()

        return observation, reward, done, info

    # compute reward at each step (with difference between observed road point and middle of the road)
    def compute_reward(self, car_state, info_collision, evaluation=False):
        # Define some constant parameters for the reward function
        THRESH_DIST = 3.7  # The maximum distance from the center of the road to compute the reward

        DISTANCE_DECAY_RATE = 0.8  # The rate at which the reward decays for the distance function
        road_points = self.reward_points

        if (info_collision.has_collided):
            return 0.0, True

        # If the car is stopped, the reward is always zero
        speed = car_state.speed
        if (speed < 3):
            return 0.0, True

        car_point = np.array(
            [car_state.kinematics_estimated.position.x_val, car_state.kinematics_estimated.position.y_val, 0])

        distance = 999

        # Compute the distance to the nearest center line
        for line in road_points:
            local_distance = 0
            length_squared = ((line[0][0] - line[1][0]) ** 2) + ((line[0][1] - line[1][1]) ** 2)
            if length_squared != 0:
                t = max(0, min(1, np.dot(car_point - line[0], line[1] - line[0]) / length_squared))
                proj = line[0] + (t * (line[1] - line[0]))
                local_distance = np.linalg.norm(proj - car_point)

            distance = min(distance, local_distance)

        reward = math.exp(-(distance * DISTANCE_DECAY_RATE))

        return reward, distance > THRESH_DIST

    # return 1 if episode is finished
    def is_done(self, car_state, info_collision, out_street):

        return 1 if info_collision.has_collided or car_state.speed < 3 or out_street else 0