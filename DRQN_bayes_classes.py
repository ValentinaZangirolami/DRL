import numpy as np
import random
import os
import time
import tensorflow as tf
import json
import cv2
from scipy.stats import t as tdist
from AirsimEnv.AirsimEnv import AirsimEnv
from AirsimEnv.bayesian import Beta, Average

TOTAL_FRAMES = 1000000         # Total number of frames to train for
EPSILON_ANNELING_FRAMES = 500000
MAX_EPISODE_LENGTH = 2000        # Maximum length of an episode (in frames).  18000 frames / 60 fps = 5 minutes
FRAMES_BETWEEN_EVAL = 50000      # Number of frames between evaluations
UPDATE_FREQ = 10000               # Number of actions chosen between updating the target network

TRACE_LENGTH = 8
DISCOUNT_FACTOR = 0.99            # Gamma, how much to discount future rewards
MIN_REPLAY_MEMORY_SIZE = 5000     # The minimum size the replay buffer must be before we start to update the agent
MEM_SIZE = 10000              # The maximum size of the replay buffer

INPUT_SHAPE = (66, 200, 3)      # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
BATCH_SIZE = 8                   # Number of samples the agent learns from at once
NUM_ACTIONS = 5

STARTING_POINTS = [(88, -1, 0.2, 1, 0, 0, 0),
                        (127.5, 45, 0.2, 0.7, 0, 0, 0.7),
                        (30, 127.3, 0.2, 1, 0, 0, 0),
                        (-59.5, 126, 0.2, 0, 0, 0, 1),
                        (-127.2, 28, 0.2, 0.7, 0, 0, 0.7),
                        (-129, -48, 0.2, 0.7, 0, 0, -0.7),
                        (-90, -128.5, 0.2, 0, 0, 0, 1),
                        (0, -86, 0.2, 0.7, 0, 0, -0.7),
                        (62, -128.3, 0.2, 1, 0, 0, 0),
                        (127, -73, 0.2, 0.7, 0, 0, -0.7)]

LOAD_REPLAY_MEMORY = True
WRITE_TENSORBOARD = True
WEIGHT_PATH = 'mnith_weights.h5'  # weights of the pre-trained network


class AirSimWrapper:

    def __init__(self, input_shape, ip, port):
        self.env = AirsimEnv(ip, port)
        self.input_shape = input_shape
        self.state = np.empty(input_shape)

    def frameProcessor(self, frame):
        # assert frame.dim == 3
        frame = frame[40:136, 0:255, 0:3]
        frame = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST)
        return frame

    def reset(self, starting_point):

        observation = self.env.reset(starting_point)
        time.sleep(0.2)
        self.env.step(0)
        speed = self.env.client.getCarState().speed
        while speed < 3.0:
            speed = self.env.client.getCarState().speed

        frame = self.frameProcessor(observation)
        self.state = frame

    def step(self, action):

        new_frame, reward, done, info = self.env.step(action)
        processed_frame = self.frameProcessor(new_frame)

        self.state = processed_frame

        return processed_frame, reward, done


class ReplayMemory:
    def __init__(self, buffer_size=MEM_SIZE):
        self.buffer = []
        self.buffer_size = buffer_size

    #add state, action, reward, next_state and terminal into buffer
    def add_experience(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0: (1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)


    #Traces of experience: Take random episode, take a random point into episode and construct a trace with length equal to trace_length
    def sample(self, batch_size=4, trace_length=8):
        sample_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sample_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point: point + trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size*trace_length, 5])

    def save(self, folder_name):
        """Save the replay buffer to a folder"""

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        np.save(folder_name + '/buffer1.npy', self.buffer[0:(len(self.buffer)+1)//2])
        np.save(folder_name + '/buffer2.npy', self.buffer[(len(self.buffer) + 1)//2:])

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.buffer = list(np.load(folder_name + '/buffer.npy', allow_pickle=True))


class Agent:
    def __init__(self, main_dqn, target_dqn, replay_memory, num_actions, input_shape,
                 batch_size=4, trace_length=TRACE_LENGTH,
                 eps_evaluation=0.0,  mu=0.0, tau=1.0, a=500.0, b=500.0, alpha=1.0, beta=1.1):

        self.main_dqn = main_dqn
        self.target_dqn = target_dqn
        self.num_actions = num_actions
        self.replay_memory = replay_memory
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.trace_length = trace_length
        self.eps_evaluation = eps_evaluation

        self.mu0, self.tau0, self.a0, self.b0 = mu, tau, a, b
        self.alpha, self.beta = alpha, beta
        self.stat = Average()
        self.post = Beta(self.alpha, self.beta)


    def calc_epsilon(self, eval=False):
        """
        Get appropriate epsilon value by first moment of beta distribution
        """
        if eval:
            return self.eps_evaluation
        else:
            post = self.post
            return post.alpha / (post.alpha + post.beta)

    def update_posterior(self, data):
        # compute epsilon
        epsilon = self.alpha / (self.alpha + self.beta)

        # update mu-hat and sigma^2-hat (sample mean and variance of the returns in D(set of previously observed returns))
        self.stat.update((1.0 - epsilon) * data[0] + epsilon * data[1])
        mu, sigma2, t = self.stat.mean, self.stat.var, self.stat.count

        # update a_t and b_t (parameters of marginal posterior distribution of the variance of the returns)
        a = self.a0 + t / 2
        b = self.b0 + t / 2 * sigma2 + t / 2 * (self.tau0 / (self.tau0 + t)) * (mu - self.mu0) * (mu - self.mu0)

        # compute e_t (under T-Student distribution)
        scale = (b / a) ** 0.5
        e_u = tdist.pdf((1.0 - epsilon) * data[0] + epsilon * data[1], df=2.0 * a, loc=data[1], scale=scale)
        e_q = tdist.pdf((1.0 - epsilon) * data[0] + epsilon * data[1], df=2.0 * a, loc=data[0], scale=scale)

        # update posterior
        self.alpha, self.beta = self.post.update(e_u, e_q)

    def get_action(self, frame_number, state, eval=False):
        """
        Query the DQN for an action given a state
        """
        eps = self.calc_epsilon(eval)

        if frame_number % 100000 == 0:
            #print("frame number: ", frame_number)
            #print("epsilon value: ", eps)
            pass

        # with chance epsilon, take a random choice
        if np.random.rand(1) < eps:
            # st_time = time.time()
            action = np.random.randint(0, self.num_actions)
            time.sleep(1 / 25)
            return action

        # query the DQN for an action
        q_values = self.main_dqn.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])))[0]
        action = q_values.argmax()
        return action

    #this function is used for calculate epsilon-BMC

    def value(self, state):
        return self.main_dqn.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])))[0]

    def update_target_network(self):
        self.target_dqn.set_weights(self.main_dqn.get_weights())

    def add_experience(self, experience):
        self.replay_memory.add_experience(experience)

    def learn(self, batch_size, trace_length, gamma, frame_number):
        """
        Sample a batch_size and use it to improve the DQN.
        Returns the loss between the predicted and target Q as a float
        """
        if len(self.replay_memory.buffer) < batch_size:
            return

        mini_batch = self.replay_memory.sample(batch_size, trace_length)

        # main DQN estimates the best action in new states
        arg_q_max = self.main_dqn.predict(np.vstack(mini_batch[:, 3])).argmax(axis=1)

        # target DQN estimates the q values for new states
        future_q_values = self.target_dqn.predict(np.vstack(mini_batch[:, 3]))
        double_q = future_q_values[range(batch_size*trace_length), arg_q_max]

        # calculate targets with Bellman equation
        # if terminal_flags == 1 (the state is terminal), target_q is equal to rewards
        target_q = mini_batch[:, 2] + (gamma * double_q * (1 - mini_batch[:, 4]))

        # use targets to calculate loss and use loss to calculate gradients
        with tf.GradientTape() as tape:
            q_values = self.main_dqn(np.vstack(mini_batch[:, 0]))

            one_hot_actions = tf.keras.utils.to_categorical(mini_batch[:, 1], self.num_actions, dtype=np.float32)
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            # mask gradients
            error = tf.square(target_q - Q)
            maskA = tf.zeros([batch_size, trace_length//2])
            maskB = tf.ones([batch_size, trace_length//2])
            mask = tf.reshape(tf.concat([maskA, maskB], 1), [-1])

            loss = tf.reduce_mean(error*mask)


        model_gradients = tape.gradient(loss, self.main_dqn.trainable_variables)
        self.main_dqn.optimizer.apply_gradients(zip(model_gradients, self.main_dqn.trainable_variables))

        return float(loss.numpy()), error

    def save(self, folder_name, **kwargs):
        """Saves the Agent and all corresponding properties into a folder
        """

        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN
        self.main_dqn.save(folder_name + '/main_dqn.h5')
        self.target_dqn.save(folder_name + '/target_dqn.h5')


        # Save replay buffer
        self.replay_memory.save(folder_name + '/replay-memory')

        # Save meta
        with open(folder_name + '/meta.json', 'w+') as f:
            f.write(json.dumps({**kwargs}))  # save replay_memory information and any other information

    def load(self, folder_name, load_replay_memory=True):
        """Load a previously saved Agent from a folder
        """

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # Load DQNs
        self.main_dqn = tf.keras.models.load_model(folder_name + '/main_dqn.h5')
        self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')

        # Load replay buffer
        if load_replay_memory:
            self.replay_memory.load(folder_name + '/replay-memory')

        # Load meta
        with open(folder_name + '/meta.json', 'r') as f:
            meta = json.load(f)
        return meta