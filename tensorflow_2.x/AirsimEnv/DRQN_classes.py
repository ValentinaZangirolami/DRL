import numpy as np
import random
import os
import time
import json
import math
from collections import deque
import cv2
from tensorflow.keras import layers, Model
import tensorflow as tf  # Keep this to manage any high-level TensorFlow functions

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
WRITE_TENSORBOARD = True

TRACE_LENGTH = 7
DISCOUNT_FACTOR = 0.99            # Gamma, how much to discount future rewards
MIN_REPLAY_MEMORY_SIZE = 100     # The minimum size the replay buffer must be before we start to update the agent
MEM_SIZE = 1000              # The maximum size of the replay buffer

# Follow 2 parameters are used to reduce hours of training phase
MIN_FRAME_START_TEST = 0   # The minimum number of frames to start the test phase
MIN_FRAME_START_SAVE = 0    # The minimum number of frames to start file saves

# Follow 2 parameters are used to optimization phase
NUM_ERROR_MASK = 3 # number of first errors to be masked
NUM_STATES_UPDATE = TRACE_LENGTH-NUM_ERROR_MASK # number of last states to be updated

INPUT_SHAPE = (66, 200, 3)      # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
BATCH_SIZE = 10                   # Number of samples the agent learns from at once

# Initial value of VDBE
EPS_INITIAL = 1

# Choice of exploitation-exploration strategy
EPS_CONST = True
VDBE = False
EPS_DESC_EPISODE = False
EPS_DESC = False

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
        # Cropping
        frame = frame[40:136, 0:255, 0:3]

        # Downsampling
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

        return self.state

    def step(self, action):

        new_frame, reward, done, info = self.env.step(action)
        processed_frame = self.frameProcessor(new_frame)

        self.state = processed_frame

        return processed_frame, reward, done

class ReplayMemory:
    def __init__(self, buffer_size, input_shape):
        self.input_shape = input_shape
        self.buffer_size = buffer_size

        # deque for efficient FIFO
        self.state = deque(maxlen=buffer_size)
        self.action = deque(maxlen=buffer_size)
        self.reward = deque(maxlen=buffer_size)
        self.next_state = deque(maxlen=buffer_size)
        self.terminal = deque(maxlen=buffer_size)

    def add_experience(self, frames, actions, rewards, next_frames, terminal):
        """
        Add a new experience to the replay memory.
        """
        self.state.append(np.array(frames, dtype=np.uint8))
        self.action.append(actions)
        self.reward.append(rewards)
        self.next_state.append(np.array(next_frames, dtype=np.uint8))
        self.terminal.append(terminal)

    def sample(self, batch_size, trace_length):
        """
        Sample a batch of experiences from the replay memory.
        """
        # Randomly sample episodes
        sample_episodes = np.random.randint(0, len(self.action), size=batch_size)
        sampled_state, sampled_action, sampled_reward, sampled_nextstate, sampled_terminal = [], [], [], [], []

        for index in sample_episodes:
            episode_state = self.state[index]
            episode_action = self.action[index]
            episode_reward = self.reward[index]
            episode_nextstate = self.next_state[index]
            episode_terminal = self.terminal[index]

            # Random point in episode
            point = np.random.randint(0, len(episode_action) + 1 - trace_length)

            # Collect traces
            sampled_state.append(episode_state[point: point + trace_length])
            sampled_action.append(episode_action[point: point + trace_length])
            sampled_reward.append(episode_reward[point: point + trace_length])
            sampled_nextstate.append(episode_nextstate[point: point + trace_length])
            sampled_terminal.append(episode_terminal[point: point + trace_length])

        # Convert lists to numpy arrays and reshape for batch processing
        sampled_state = np.reshape(np.array(sampled_state), [batch_size * trace_length, *self.input_shape])
        sampled_action = np.array(sampled_action).reshape(batch_size * trace_length)
        sampled_reward = np.array(sampled_reward).reshape(batch_size * trace_length)
        sampled_nextstate = np.reshape(np.array(sampled_nextstate), [batch_size * trace_length, *self.input_shape])
        sampled_terminal = np.array(sampled_terminal).reshape(batch_size * trace_length)

        return sampled_state, sampled_reward, sampled_action, sampled_nextstate, sampled_terminal

    def save(self, file_path, compressed=False):
        """
        Save the replay memory to a file. The data is first converted to lists,
        then saved as numpy arrays.
        """
        # Convert deques to lists
        data = {
            'state': list(self.state),
            'action': list(self.action),
            'reward': list(self.reward),
            'next_state': list(self.next_state),
            'terminal': list(self.terminal)
        }

        # Convert lists to numpy arrays
        data = {key: np.array(value, dtype=object) for key, value in data.items()}

        # Save data in compressed format or regular format
        if compressed:
            np.savez_compressed(file_path, **data)
        else:
            np.savez(file_path, **data)

    def load(self, file_path, compressed=False):
        """
        Load the replay memory from a file. The data is loaded as numpy arrays and then
        converted back to deques.
        """
        # Load data from file
        if compressed:
            data = np.load(file_path, allow_pickle=True)
        else:
            data = np.load(file_path, allow_pickle=True)

        # Convert numpy arrays to lists and then to deques
        self.state = deque(data['state'].tolist(), maxlen=self.buffer_size)
        self.action = deque(data['action'].tolist(), maxlen=self.buffer_size)
        self.reward = deque(data['reward'].tolist(), maxlen=self.buffer_size)
        self.next_state = deque(data['next_state'].tolist(), maxlen=self.buffer_size)
        self.terminal = deque(data['terminal'].tolist(), maxlen=self.buffer_size)


class QNetwork(Model):
    def __init__(self, h_size, rnn_cell, num_action, num_error_mask, num_states_update, batch_size_training):
        super(QNetwork, self).__init__()

        # Define convolutional layers
        self.conv1 = layers.Conv2D(32, (8, 8), strides=(4, 4), padding='valid', activation='relu')
        self.conv2 = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='valid', activation='relu')
        self.conv3 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu')

        # Define RNN layer
        self.rnn = layers.RNN(rnn_cell, return_sequences=True, return_state=True, dtype=tf.float32)

        # Value and Advantage streams
        self.streamA = layers.Dense(h_size // 2, dtype=tf.float32)
        self.streamV = layers.Dense(h_size // 2, dtype=tf.float32)
        self.AW = layers.Dense(num_action, dtype=tf.float32)
        self.VW = layers.Dense(1, dtype=tf.float32)

        # Mask for TD error calculation
        self.num_error_mask = num_error_mask
        self.num_states_update = num_states_update
        self.num_tot = self.num_error_mask + self.num_states_update
        self.batch_size_training = batch_size_training
        self.num_actions = num_action


    def call(self, inputs, initial_state=None, training=False):
        # Process through convolutional layers
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten and reshape for RNN input
        x = tf.keras.layers.Flatten()(x)

        # Determine batch_size and trace_length based on input shape using TensorFlow operations
        is_training = tf.equal(tf.shape(x)[0], self.batch_size_training * self.num_tot)

        # Use tf.cond to select values based on condition
        batch_size = tf.cond(is_training, lambda: self.batch_size_training, lambda: 1)
        trace_length = tf.cond(is_training, lambda: self.num_tot, lambda: 1)

        # Reshape to [batch_size, trace_length, flattened_size] for RNN
        x = tf.reshape(x, [batch_size, trace_length, x.shape[-1]])

        # Process through RNN layer
        x, state, last_cell = self.rnn(x, initial_state=initial_state)
        x = tf.reshape(x, [-1, x.shape[-1]])

        # Advantage and Value streams
        advantage = self.AW(self.streamA(x))
        value = self.VW(self.streamV(x))

        # Combine Value and Advantage to get Q-values
        q_out = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

        return q_out, (state, last_cell)

    def compute_loss(self, q_out, actions, target_q, batch_size):

        maskA = tf.zeros([batch_size, self.num_error_mask], dtype=tf.float32)  # Mask for errors to be ignored
        maskB = tf.ones([batch_size, self.num_states_update], dtype=tf.float32)  # Mask for errors to be updated

        actions_onehot = tf.one_hot(actions, self.num_actions, dtype=tf.float32)
        q = tf.reduce_sum(tf.math.multiply(q_out, actions_onehot), axis=1)
        td_error = tf.square(target_q - q)

        # Apply mask
        mask = tf.concat([maskA, maskB], 1)
        mask = tf.reshape(mask, [-1])
        loss = tf.reduce_mean(td_error * mask)

        return loss


class Agent:
    def __init__(self, main_drqn, target_drqn, replay_memory, num_actions, input_shape,
                 batch_size=10, tau_soft=0.1, eps_initial=1, eps_final=0.1, eps_final_frame=0.01, trace_length=TRACE_LENGTH,
                 eps_evaluation=0.0, eps_annealing_frames=400000, eps_annealing_episode=12000, replay_memory_start_size=50000, max_frames=700000, eps_constant=0.05, delta_vdbe=0.2, sigma_vdbe=1.0, eps_vdbe=0.9):

        self.main_drqn = main_drqn
        self.target_drqn = target_drqn
        self.num_actions = num_actions
        self.replay_memory = replay_memory
        self.replay_memory_start_size = replay_memory_start_size
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.trace_length = trace_length

        # parameters of Epsilon-vdbe
        self.delta_vdbe = delta_vdbe
        self.sigma_vdbe = sigma_vdbe
        self.list_vdbe = []

        # Epsilon information
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.eps_annealing_episode = eps_annealing_episode
        self.eps_constant = eps_constant

        # Slopes and intercepts for exploration decrease
        # (Credit to Fabio M. Graetz for this and calculating epsilon based on frame number)
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_memory_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (
            self.max_frames - self.eps_annealing_frames - self.replay_memory_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2 * self.max_frames

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    def calc_epsilon(self, frame_number, eval=False, eps_const=False, vdbe=False):
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
        else:
            # Descending eps-greedy (based on frame number)
            if frame_number < self.replay_memory_start_size:
                return self.eps_initial
            elif frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
                return self.slope * frame_number + self.intercept
            elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
                return self.slope_2 * frame_number + self.intercept_2

    def update_vdbe(self, td_error):
        coeff = math.exp(-abs(td_error)/self.sigma_vdbe)
        # Boltzmann distribution
        f = (1.0 - coeff) / (1.0 + coeff)
        # update epsilon
        fin = len(self.list_vdbe)
        self.list_vdbe.append(self.delta_vdbe * f + (1.0 - self.delta_vdbe) * self.list_vdbe[fin-1])

    def get_action(self, frame_number, state, state_in, eval=False, eps_const=True, vdbe=False):
        """
        Query the DRQN for an action given a state
        """

        # Normalize the input state
        normalized_state = state / 255.0

        # Add batch and time dimensions (since we're processing a single step in the sequence)
        normalized_state = tf.convert_to_tensor(normalized_state, dtype=tf.float32)
        normalized_state = tf.expand_dims(normalized_state, axis=0)  # Add batch dimension
        normalized_state = tf.expand_dims(normalized_state, axis=1)  # Add time dimension


        eps = self.calc_epsilon(frame_number, eval, eps_const, vdbe)

        if frame_number % 100000 == 0:
            #print("frame number: ", frame_number)
            #print("epsilon value: ", eps)
            pass

        # with chance epsilon, take a random choice
        if np.random.rand(1) < eps:
            # st_time = time.time()
            # Query the model for the next RNN state
            _, state1 = self.main_drqn(normalized_state, initial_state=state_in, training=False)
            action = np.random.randint(0, self.num_actions)
            time.sleep(1 / 25)
            return action, state1

        # Query the model for action and the next RNN state
        q_values, state1 = self.main_drqn(normalized_state, initial_state=state_in, training=False)

        # Choose the action with the highest Q-value
        action = tf.argmax(q_values, axis=1)

        # Return the selected action and the updated RNN state
        return int(action.numpy()[0]), state1

    # 'value' is used to calculate epsilon-BMC and VDBE
    def value(self, state, state_in):
        # Normalize the input state
        normalized_state = state / 255.0

        # Add batch and time dimensions (since we're processing a single step in the sequence)
        normalized_state = tf.expand_dims(normalized_state, axis=0)  # Add batch dimension
        normalized_state = tf.expand_dims(normalized_state, axis=1)  # Add time dimension

        # Query the model for Q-value
        q_values, _ = self.main_drqn(normalized_state, initial_state=state_in, training=False)
        return q_values[0]

    # add experience to buffer
    def add_experience(self, frames, actions, rewards, next_frames, terminal):
        self.replay_memory.add_experience(frames, actions, rewards, next_frames, terminal)

    def learn(self, batch_size, trace_length, gamma, state_train, frame_number):
        """
        Sample a batch_size and use it to improve the DRQN.
        Returns the loss between the predicted and target Q as a float
        """
        if len(self.replay_memory.action) < batch_size:
            return

        # take sampled experience
        state, reward, action, next_state, terminal = self.replay_memory.sample(batch_size, trace_length)

        # Normalize the states by dividing by 255.0 (assuming pixel values as inputs)
        state = tf.convert_to_tensor(state / 255.0, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state / 255.0, dtype=tf.float32)

        # Main DQN estimates the best action in new states
        arg_q_max = tf.argmax(self.main_drqn(state, training=False)[0], axis=1)
        arg_q_max = tf.cast(arg_q_max, dtype=tf.int32)

        # Target DQN estimates the q values for new states
        future_q_values, _ = self.target_drqn(next_state, initial_state=state_train, training=False)

        # doubleQ
        indices = tf.range(batch_size * trace_length, dtype=tf.int32)
        double_q = tf.gather_nd(future_q_values, tf.stack([indices, arg_q_max], axis=1))

        # Calculate targets with Bellman equation
        target_q = reward + gamma * double_q * (1 - terminal)

        with tf.GradientTape() as tape:
            # Get the current Q values from the main DRQN
            q_out, _ = self.main_drqn(state, initial_state=state_train, training=True)

            # Compute the loss
            loss = self.main_drqn.compute_loss(q_out, action, target_q, batch_size)

        # Compute gradients
        gradients = tape.gradient(loss, self.main_drqn.trainable_variables)

        # Apply the gradients using the optimizer
        self.optimizer.apply_gradients(zip(gradients, self.main_drqn.trainable_variables))

        return float(loss)

    def update_target_network(self, tau=0.001):
        for main_var, target_var in zip(self.main_drqn.trainable_variables, self.target_drqn.trainable_variables):
            target_var.assign(tau * main_var + (1 - tau) * target_var)

    def save(self, folder_name, vdbe, **kwargs):
        """
        Saves the Agent and all corresponding properties into a folder
        """

        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save the main and target DRQN models' weights using the SavedModel format
        self.main_drqn.save(os.path.join(folder_name, 'main_drqn_model'))
        self.target_drqn.save(os.path.join(folder_name, 'target_drqn_model'))

        # Save replay buffer in compressed format
        self.replay_memory.save(os.path.join(folder_name, 'replay_memory.npz'), compressed=True)

        # Save meta information compactly
        with open(os.path.join(folder_name, 'meta.json'), 'w') as f:
            meta_info = {'epsilon': self.list_vdbe} if vdbe else {}
            f.write(json.dumps({**meta_info, **kwargs}, separators=(',', ':')))  # Compact JSON format

    def load(self, folder_name, vdbe, load_replay_memory=True):
        """Load a previously saved Agent from a folder
        """

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # Load the main and target DRQN models' weights from SavedModel format
        self.main_drqn = tf.keras.models.load_model(os.path.join(folder_name, 'main_drqn_model'))
        self.target_drqn = tf.keras.models.load_model(os.path.join(folder_name, 'target_drqn_model'))

        # Load replay buffer from compressed format
        if load_replay_memory:
            self.replay_memory.load(os.path.join(folder_name, 'replay_memory.npz'), compressed=True)

        # Load meta information
        with open(os.path.join(folder_name, 'meta.json'), 'r') as f:
            meta = json.load(f)

        if vdbe:
            self.list_vdbe = meta['epsilon']

        return meta