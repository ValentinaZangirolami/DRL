import numpy as np
import random
import os
import time
import tensorflow as tf
import json
import math
import cv2
from scipy.stats import t as tdist
from AirsimEnv.bayesian import Beta, Average
import tf_slim as slim
#Choice the number of steering angle
NUM_ACTIONS = 5
if NUM_ACTIONS == 5:
    from AirsimEnv.AirsimEnv import AirsimEnv
elif NUM_ACTIONS == 9:
    from AirsimEnv.AirsimEnv_9actions import AirsimEnv

TOTAL_FRAMES = 1000000           # Total number of frames to train for
TOTAL_EPISODE = 24000
EPSILON_ANNELING_FRAMES = 400000
MAX_EPISODE_LENGTH = 2000        # Maximum length of an episode (in frames).  18000 frames / 60 fps = 5 minutes (when process is based on episode, this par. is equal to 10000 (Episode to infinity))
FRAMES_BETWEEN_EVAL = 50000      # Number of frames between evaluations

TRACE_LENGTH = 10
DISCOUNT_FACTOR = 0.99            # Gamma, how much to discount future rewards
MIN_REPLAY_MEMORY_SIZE = 999     # The minimum size the replay buffer must be before we start to update the agent
MEM_SIZE = 1000              # The maximum size of the replay buffer

# Follow 2 parameters are used to reduce hours of training phase
MIN_FRAME_START_TEST = 0   # The minimum number of frames to start the test phase
MIN_FRAME_START_SAVE = 0    # The minimum number of frames to start file saves

# Follow 2 parameters are used to optimization phase
NUM_ERROR_MASK = 7 #number of first errors to be masked
NUM_STATES_UPDATE = TRACE_LENGTH-NUM_ERROR_MASK # number of last states to be updated

INPUT_SHAPE = (66, 200, 3)      # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
BATCH_SIZE = 10                   # Number of samples the agent learns from at once

# Parameters of EPS-BMC
ALPHA = 25.0
BETA = 25.01
# Initial value of VDBE
EPS_INITIAL = 1

# Choice of exploitation-exploration strategy
BAYES = False  # Take bayes=T only with eps_greedy=T
EPS_CONST = True
VDBE = False
EPS_DESC_EPISODE = False
EPS_DESC = False
SOFTMAX = False #If Softmax=True, others strategy=false
EPS_GREEDY = False #if eps_greedy=True, you also must select the eps greedy method (Ex: bayes, vdbe or others strategies)
MBE = True #Max Boltzmann Exploration (Please, attention: if you chose MBE, you must select eps greedy strategy (VDBE-Softmax:MBE=True and VDBE=True))


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
    def __init__(self, buffer_size, input_shape):

        self.input_shape = input_shape
        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.terminal = []
        self.buffer_size = buffer_size

    #add state, action, reward, next_state and terminal into buffer
    def add_experience(self, frames, actions, rewards, next_frames, terminal):
        if len(self.action) + 1 >= self.buffer_size:
            self.state[0: (1 + len(self.state)) - self.buffer_size] = []
            self.action[0: (1 + len(self.action)) - self.buffer_size] = []
            self.reward[0: (1 + len(self.reward)) - self.buffer_size] = []
            self.next_state[0: (1 + len(self.next_state)) - self.buffer_size] = []
            self.terminal[0: (1 + len(self.terminal)) - self.buffer_size] = []
        self.state.append(frames)
        self.action.append(actions)
        self.reward.append(rewards)
        self.next_state.append(next_frames)
        self.terminal.append(terminal)

    #Traces of experience: Take random episode, take a random point into episode and construct a trace with length equal to trace_length
    def sample(self, batch_size=BATCH_SIZE, trace_length=TRACE_LENGTH):
        # Sample of indexes for each component of buffer
        sample_episodes = np.random.randint(0, len(self.action), size=batch_size)
        sampled_action = []
        sampled_reward = []
        sampled_state = []
        sampled_nextstate = []
        sampled_terminal = []
        for index in sample_episodes:
            episode_state, episode_action, episode_reward = self.state[index], self.action[index], self.reward[index]
            episode_nextstate, episode_terminal = self.next_state[index], self.terminal[index]
            # Random point in episode
            point = np.random.randint(0, len(episode_action) + 1 - trace_length)
            sampled_action.append(episode_action[point: point + trace_length])
            sampled_reward.append(episode_reward[point: point + trace_length])
            sampled_state.append(episode_state[point: point + trace_length])
            sampled_nextstate.append(episode_nextstate[point: point + trace_length])
            sampled_terminal.append(episode_terminal[point: point + trace_length])
        sampled_state = np.reshape(np.array(sampled_state), [batch_size * trace_length,  self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        sampled_reward = np.reshape(np.array(sampled_reward), [batch_size*trace_length])
        sampled_nextstate = np.reshape(np.array(sampled_nextstate), [batch_size * trace_length, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        sampled_action = np.reshape(np.array(sampled_action), [batch_size * trace_length])
        sampled_terminal = np.reshape(np.array(sampled_terminal), [batch_size * trace_length])
        return sampled_state, sampled_reward, sampled_action, sampled_nextstate, sampled_terminal

    def save(self, folder_name):
        """Save the replay buffer to a folder"""

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        np.save(folder_name + '/state.npy', self.state)
        np.save(folder_name + '/action.npy', self.action)
        np.save(folder_name + '/reward.npy', self.reward)
        np.save(folder_name + '/nextstate.npy', self.next_state)
        np.save(folder_name + '/terminal.npy', self.terminal)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.state = list(np.load(folder_name + '/state.npy', allow_pickle=True))
        self.action = list(np.load(folder_name + '/action.npy', allow_pickle=True))
        self.reward = list(np.load(folder_name + '/reward.npy', allow_pickle=True))
        self.next_state = list(np.load(folder_name + '/nextstate.npy', allow_pickle=True))
        self.terminal = list(np.load(folder_name + '/terminal.npy', allow_pickle=True))


class Qnetwork:
    def __init__(self, h_size, rnn_cell, myScope, num_action, num_error_mask, num_states_update):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = tf.compat.v1.placeholder(shape=[None, 66, 200, 3], dtype=tf.float32)
        self.conv1 = slim.convolution2d(inputs=self.scalarInput, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding='VALID',biases_initializer=None, scope=myScope +'_conv1')
        self.conv2 = slim.convolution2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID', biases_initializer=None, scope=myScope +'_conv2')
        self.conv3 = slim.convolution2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID', biases_initializer=None, scope=myScope +'_conv3')
        self.trainLength = tf.compat.v1.placeholder(dtype=tf.int32)
        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levles.
        self.batch_size = tf.compat.v1.placeholder(dtype=tf.int32, shape=[])
        self.convFlat = tf.compat.v1.reshape(slim.flatten(self.conv3), [self.batch_size, self.trainLength, slim.flatten(self.conv3).shape[1]])
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.compat.v1.nn.dynamic_rnn(inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope + '_rnn')
        self.rnn = tf.compat.v1.reshape(self.rnn, shape=[-1, h_size])
        # The output from the recurrent player is then split into separate Value and Advantage streams
        self.streamA, self.streamV = tf.compat.v1.split(self.rnn, 2, 1)
        self.AW = tf.compat.v1.Variable(tf.compat.v1.random_normal([h_size // 2, num_action]))
        self.VW = tf.compat.v1.Variable(tf.compat.v1.random_normal([h_size // 2, 1]))
        self.Advantage = tf.compat.v1.matmul(self.streamA, self.AW)
        self.Value = tf.compat.v1.matmul(self.streamV, self.VW)

        self.salience = tf.compat.v1.gradients(self.Advantage, self.scalarInput)
        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.compat.v1.subtract(self.Advantage, tf.compat.v1.reduce_mean(self.Advantage, axis=1, keep_dims=True))

        self.predict = tf.compat.v1.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.compat.v1.one_hot(self.actions, num_action, dtype=tf.float32)

        self.Q = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.compat.v1.square(self.targetQ - self.Q)

        # In order to only propogate accurate gradients through the network, we will mask the first
        # half of the losses for each trace as per Lample & Chatlot 2016
        self.maskA = tf.compat.v1.zeros([self.batch_size, num_error_mask])
        self.maskB = tf.compat.v1.ones([self.batch_size, num_states_update])
        self.mask = tf.compat.v1.concat([self.maskA, self.maskB], 1)
        self.mask = tf.compat.v1.reshape(self.mask, [-1])
        self.loss = tf.compat.v1.reduce_mean(self.td_error * self.mask)

        self.trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class Agent:
    def __init__(self, main_drqn, target_drqn, replay_memory, beta, average, num_actions, input_shape,
                 batch_size=10, tau_soft=0.1, eps_initial=1, eps_final=0.1, eps_final_frame=0.01, trace_length=TRACE_LENGTH,
                 eps_evaluation=0.0, eps_annealing_frames=400000, eps_annealing_episode=12000, replay_memory_start_size=50000, episode_start_size=1000, max_frames=700000, max_episode=24000, mu=0.0, tau=1.0, a=250.0, b=250.0, eps_constant=0.05, delta_vdbe=0.2, sigma_vdbe=1.0, eps_vdbe=0.9):

        self.main_drqn = main_drqn
        self.target_drqn = target_drqn
        self.num_actions = num_actions
        self.replay_memory = replay_memory
        self.replay_memory_start_size = replay_memory_start_size
        self.episode_start_size = episode_start_size
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.max_episode = max_episode
        self.trace_length = trace_length

        # parameters of Epsilon-bmc
        self.mu0, self.tau0, self.a0, self.b0 = mu, tau, a, b
        self.stat = average
        self.post = beta
        self.eps_bmc = []
        self.eps_constant = eps_constant

        # parameters of Epsilon-vdbe
        self.delta_vdbe = delta_vdbe
        self.sigma_vdbe = sigma_vdbe
        self.list_vdbe = []

        #parameter of softmax
        self.tau = tau_soft

        #Lists for softmax
        self.q_values_list = []
        self.probs_list = []

        # Epsilon information
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.eps_annealing_episode = eps_annealing_episode

        # Slopes and intercepts for exploration decrease
        # (Credit to Fabio M. Graetz for this and calculating epsilon based on frame number)
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_memory_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (
            self.max_frames - self.eps_annealing_frames - self.replay_memory_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2 * self.max_frames

        # Slopes and intercepts for exploration decrease with process based on episode
        self.slope_ep = -(self.eps_initial - self.eps_final) / self.eps_annealing_episode
        self.intercept_ep = self.eps_initial - self.slope * self.episode_start_size
        self.slope_ep_2 = -(self.eps_final - self.eps_final_frame) / (
                self.max_episode - self.eps_annealing_episode - self.episode_start_size)
        self.intercept_ep_2 = self.eps_final_frame - self.slope_2 * self.max_episode

    def calc_epsilon(self, frame_number, episode_number, eval=False, bayes=False, eps_const=False, vdbe=False, eps_episode=False):
        """
        Get the appropriate epsilon value based on choice of strategy
        """
        if eval==True:
            # epsilon for evaluation phase
            return self.eps_evaluation
        elif eps_const==True:
            # eps-constant
            if len(self.replay_memory.action) > MIN_REPLAY_MEMORY_SIZE:
                return self.eps_constant
            else:
                return 1
        elif vdbe==True:
            # Adaptive eps-greedy vdbe
            fin = len(self.list_vdbe)
            return self.list_vdbe[fin-1]
        elif bayes==True:
            # expected value of Beta distribution (Adaptive eps-bmc)
            if len(self.replay_memory.action) > MIN_REPLAY_MEMORY_SIZE:
                post = self.post
                length_list = len(post.alpha)
                alpha = post.alpha[length_list-1]
                beta = post.beta[length_list-1]
                self.eps_bmc.append(alpha/(alpha + beta))
                return alpha / (alpha + beta)
            else:
                # Epsilon constant(=1) for the first phase: when buffer collects MEM_SIZE episodes, eps-bmc starts updating
                self.eps_bmc.append(1)
                return 1
        elif eps_episode==True:
            # Descending eps-greedy (based on episode number)
            if episode_number < self.episode_start_size:
                return self.eps_initial
            elif episode_number >= self.episode_start_size and episode_number< self.episode_start_size + self.eps_annealing_episode:
                return self.slope_ep * episode_number + self.intercept_ep
            elif episode_number >= self.episode_start_size + self.eps_annealing_episode:
                return self.slope_ep_2 * episode_number + self.intercept_ep_2
        else:
            # Descending eps-greedy (based on frame number)
            if frame_number < self.replay_memory_start_size:
                return self.eps_initial
            elif frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
                return self.slope * frame_number + self.intercept
            elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
                return self.slope_2 * frame_number + self.intercept_2

    def update_posterior(self, data):
        length_list = len(self.post.alpha)
        alpha = self.post.alpha[length_list - 1]
        beta = self.post.beta[length_list - 1]
        # compute epsilon
        epsilon = alpha / (alpha + beta)
        # (Credit to Michael Gimelfarb for this calculation)

        # update mu-hat and sigma^2-hat (sample mean and variance of the returns in D(set of previously observed returns))
        self.stat.update((1.0 - epsilon) * data[0] + epsilon * data[1])
        mu, t = self.stat.mean, self.stat.count
        last_var = len(self.stat.var)
        sigma2 = self.stat.var[last_var-1]

        # update a_t and b_t (parameters of marginal posterior distribution of the variance of the returns)
        a = self.a0 + t / 2
        b = self.b0 + t / 2 * sigma2 + t / 2 * (self.tau0 / (self.tau0 + t)) * (mu - self.mu0) * (mu - self.mu0)

        # compute e_t (under T-Student distribution)
        scale = (b / a) ** 0.5
        e_u = tdist.pdf((1.0 - epsilon) * data[0] + epsilon * data[1], df=2.0 * a, loc=data[1], scale=scale)
        e_q = tdist.pdf((1.0 - epsilon) * data[0] + epsilon * data[1], df=2.0 * a, loc=data[0], scale=scale)

        # update posterior
        self.post.update(e_u, e_q)

    def update_vdbe(self, td_error):
        coeff = math.exp(-abs(td_error)/self.sigma_vdbe)
        # Boltzmann distribution
        f = (1.0 - coeff) / (1.0 + coeff)
        # update epsilon
        fin = len(self.list_vdbe)
        self.list_vdbe.append(self.delta_vdbe * f + (1.0 - self.delta_vdbe) * self.list_vdbe[fin-1])

    def get_action(self, frame_number, episode_number, state, state_in, session, eval=False, bayes=False, eps_const=False, vdbe=False, eps_episode=False, eps_greedy=False, softmax=False, mbe=False):
        """
        Query the DRQN for an action given a state
        """
        if eps_greedy == True:
            eps = self.calc_epsilon(frame_number, episode_number, eval, bayes, eps_const, vdbe, eps_episode)

            if frame_number % 100000 == 0:
                #print("frame number: ", frame_number)
                #print("epsilon value: ", eps)
                pass

            # with chance epsilon, take a random choice
            if np.random.rand(1) < eps:
                # st_time = time.time()
                state1 = session.run(self.main_drqn.rnn_state, feed_dict={self.main_drqn.scalarInput: [state / 255.0], self.main_drqn.trainLength: 1, self.main_drqn.state_in: state_in, self.main_drqn.batch_size: 1})
                action = np.random.randint(0, self.num_actions)
                time.sleep(1 / 25)
                return action, state1

        elif softmax == True:
            Q, state1 = session.run([self.main_drqn.Qout, self.main_drqn.rnn_state], feed_dict={self.main_drqn.scalarInput: [state / 255.0], self.main_drqn.trainLength: 1, self.main_drqn.state_in: state_in, self.main_drqn.batch_size: 1})
            self.q_values_list.append(Q[0])
            total = sum([np.exp(val / self.tau) for val in Q[0]])
            probs = [np.exp(val / self.tau) / total for val in Q[0]]

            self.probs_list.append(probs)
            threshold = random.random()
            cumulative_prob = 0.0
            for i in range(len(probs)):
                cumulative_prob += probs[i]
                if (cumulative_prob > threshold):
                    return i, state1


        elif mbe==True:
            eps = self.calc_epsilon(frame_number, episode_number, eval, bayes, eps_const, vdbe, eps_episode)
            if np.random.rand(1) < eps:
                Q, state1 = session.run([self.main_drqn.Qout, self.main_drqn.rnn_state],
                                         feed_dict={self.main_drqn.scalarInput: [state / 255.0],
                                                    self.main_drqn.trainLength: 1, self.main_drqn.state_in: state_in,
                                                    self.main_drqn.batch_size: 1})
                total = sum([np.exp(val / self.tau) for val in Q[0]])
                probs = [np.exp(val / self.tau) / total for val in Q[0]]

                threshold = random.random()
                cumulative_prob = 0.0
                for i in range(len(probs)):
                    cumulative_prob += probs[i]
                    if (cumulative_prob > threshold):
                        return i, state1

        elif eval==True:
            action, state1, qvalues = session.run([self.main_drqn.predict, self.main_drqn.rnn_state, self.main_drqn.Qout],
                                         feed_dict={self.main_drqn.scalarInput: [state / 255.0],
                                                    self.main_drqn.trainLength: 1, self.main_drqn.state_in: state_in,
                                                    self.main_drqn.batch_size: 1})
            return action[0], state1, qvalues[0]
        # query the DRQN for an action
        action, state1 = session.run([self.main_drqn.predict, self.main_drqn.rnn_state], feed_dict={self.main_drqn.scalarInput: [state / 255.0], self.main_drqn.trainLength: 1, self.main_drqn.state_in: state_in, self.main_drqn.batch_size: 1})
        return action[0], state1

    # 'value' is used to calculate epsilon-BMC and VDBE
    def value(self, state, state_in, session):
        q_values = session.run(self.main_drqn.Qout, feed_dict={self.main_drqn.scalarInput: [state / 255.0], self.main_drqn.trainLength: 1, self.main_drqn.state_in: state_in, self.main_drqn.batch_size: 1})
        return q_values[0]

    # add experience to buffer
    def add_experience(self, frames, actions, rewards, next_frames, terminal):
        self.replay_memory.add_experience(frames, actions, rewards, next_frames, terminal)

    def learn(self, batch_size, trace_length, gamma, state_train, session, frame_number):
        """
        Sample a batch_size and use it to improve the DRQN.
        Returns the loss between the predicted and target Q as a float
        """
        if len(self.replay_memory.action) < batch_size:
            return
        # take sampled experience
        state, reward, action, next_state, terminal = self.replay_memory.sample(batch_size, trace_length)


        # main DQN estimates the best action in new states
        arg_q_max = session.run(self.main_drqn.predict, feed_dict={self.main_drqn.scalarInput: next_state/255.0, self.main_drqn.trainLength: trace_length, self.main_drqn.state_in: state_train, self.main_drqn.batch_size: batch_size})

        # target DQN estimates the q values for new states
        future_q_values = session.run(self.target_drqn.Qout, feed_dict={self.target_drqn.scalarInput: next_state/255.0 , self.target_drqn.trainLength: trace_length, self.target_drqn.state_in: state_train, self.target_drqn.batch_size: batch_size})
        double_q = future_q_values[range(batch_size * trace_length), arg_q_max]

        # calculate targets with Bellman equation
        # if terminal_flags == 1 (the state is terminal), target_q is equal to rewards
        target_q = reward + (gamma * double_q * (1 - terminal))

        # use targets to calculate loss and use loss to calculate gradients
        loss, error, _ = session.run([self.main_drqn.loss, self.main_drqn.td_error, self.main_drqn.updateModel], feed_dict={self.main_drqn.scalarInput: state/255.0, self.main_drqn.targetQ: target_q, self.main_drqn.actions: action, self.main_drqn.trainLength: trace_length, self.main_drqn.state_in: state_train, self.main_drqn.batch_size: batch_size})
        return float(loss), error

    def save(self, folder_name, softmax, bayes, vdbe, **kwargs):
        """
        Saves the Agent and all corresponding properties into a folder
        """

        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)


        # Save replay buffer
        self.replay_memory.save(folder_name + '/replay-memory')

        if softmax==True:
            np.save(folder_name + '/qvalues.npy', self.q_values_list)
            np.save(folder_name + '/probs.npy', self.probs_list)

        # Save meta
        with open(folder_name + '/meta.json', 'w+') as f:
            if bayes == True:
                f.write(json.dumps({**{'epsilon': self.eps_bmc, 'alpha': self.post.alpha, 'beta': self.post.beta, 'count_eps': self.stat.count, 'var_eps': self.stat.var, 'm2_eps': self.stat.m2, 'mean_eps': self.stat.mean}, **kwargs}))
            elif vdbe == True:
                f.write(json.dumps({**{'epsilon': self.list_vdbe}, **kwargs}))# save replay_memory information and any other information
            else:
                f.write(json.dumps({**kwargs}))

    def load(self, folder_name, softmax, bayes, vdbe, load_replay_memory=True):
        """Load a previously saved Agent from a folder
        """

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')


        # Load replay buffer
        if load_replay_memory:
            self.replay_memory.load(folder_name + '/replay-memory')

        if softmax==True:
            self.q_values_list = list(np.load(folder_name + '/qvalues.npy', allow_pickle=True))
            self.probs_list = list(np.load(folder_name + '/probs.npy', allow_pickle=True))

        # Load meta
        with open(folder_name + '/meta.json', 'r') as f:
            meta = json.load(f)

        if bayes == True:
            self.eps_bmc = meta['epsilon']
            self.post.alpha = meta['alpha']
            self.post.beta = meta['beta']
            self.stat.count = meta['count_eps']
            self.stat.var = meta['var_eps']
            self.stat.m2 = meta['m2_eps']
            self.stat.mean = meta['mean_eps']
        if vdbe == True:
            self.list_vdbe = meta['epsilon']

        return meta