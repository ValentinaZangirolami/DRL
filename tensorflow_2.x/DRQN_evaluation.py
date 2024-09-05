import numpy as np
import os
import time
import pandas as pd
import random
import rootpath
import tensorflow as tf
from AirsimEnv.DRQN_classes import Agent, AirSimWrapper, QNetwork
from AirsimEnv.DRQN_classes import (INPUT_SHAPE, NUM_ACTIONS, NUM_STATES_UPDATE, NUM_ERROR_MASK, BATCH_SIZE)

def conf_dir(env_key, default_value):
    p = os.path.expanduser(os.getenv(env_key, default_value))
    return rootpath.detect(__file__, "^.git$") + p[1:] if p.startswith("./") else p

DATA_HOME = conf_dir('PC_DATA_HOME', "./data/ext/home")
DATA_HOST = conf_dir('PC_DATA_HOST', "./data/ext/host")
DATA_USER = conf_dir('PC_DATA_USER', "./data/ext/user")
DATA_DESK = conf_dir('PC_DATA_DESK', "~/Desktop")

IP = "127.0.0.1"
PORT = 41451

TRAIN_STARTING_POINTS = [(88, -1, 0.2, 1, 0, 0, 0),
                         (127.5, 45, 0.2, 0.7, 0, 0, 0.7),
                         (30, 127.3, 0.2, 1, 0, 0, 0),
                         (-59.5, 126, 0.2, 0, 0, 0, 1),
                         (-127.2, 28, 0.2, 0.7, 0, 0, 0.7),
                         (-129, -48, 0.2, 0.7, 0, 0, -0.7),
                         (-90, -128.5, 0.2, 0, 0, 0, 1),
                         (0, -86, 0.2, 0.7, 0, 0, -0.7),
                         (62, -128.3, 0.2, 1, 0, 0, 0),
                         (127, -73, 0.2, 0.7, 0, 0, -0.7)]

TEST_STARTING_POINTS = [(0.5, 44, 0.2, 0.7, 0, 0, 0.7),
                        (-75, -0.8, 0.2, 0, 0, 0, 1),
                        (-128.2, 45, 0.2, 0.7, 0, 0, 0.7),
                        (-0.5, -20, 0.2, 0.7, 0, 0, -0.7),
                        (127, -38, 0.2, 0.7, 0, 0, 0.7),
                        (-6, 126.5, 0, 0, 0, 0, 1),
                        (22, -127.5, 0.2, 1, 0, 0, 0),
                        (126.8, 15, 0.2, 0.7, 0, 0, -0.7),
                        (-127.2, 16, 0.2, 0.7, 0, 0, -0.7),
                        (-27, 0, 0.2, 1, 0, 0, 0)]

h_size = 512
EVALUATION_DURING_TRAINING = False

def evaluation_agent(path, num_evaluation, h_size, starting_points):
    df = pd.DataFrame(columns=["point", "reward", "time (s)", "frame", "list of rewards", "list of qvalues", "action"])
    airsim_wrapper = AirSimWrapper(ip=IP, port=PORT, input_shape=INPUT_SHAPE)
    rnn_cell = tf.keras.layers.LSTMCell(units=h_size)
    rnn_cell_T = tf.keras.layers.LSTMCell(units=h_size)
    mainQN = QNetwork(h_size, rnn_cell, num_action=NUM_ACTIONS, num_error_mask=NUM_ERROR_MASK,
                      num_states_update=NUM_STATES_UPDATE, batch_size_training=BATCH_SIZE)
    targetQN = QNetwork(h_size, rnn_cell_T, num_action=NUM_ACTIONS, num_error_mask=NUM_ERROR_MASK,
                        num_states_update=NUM_STATES_UPDATE, batch_size_training=BATCH_SIZE)

    # Replay memory is not needed for evaluation
    replay_memory = None
    agent = Agent(mainQN, targetQN, replay_memory, num_actions=NUM_ACTIONS, input_shape=INPUT_SHAPE)

    # Load model from the provided path
    print('Loading Model...')
    agent.load(path, vdbe=False, load_replay_memory=False)

    for point in starting_points:
        print("Evaluation from: ", point)
        terminal = True
        state_in = (tf.zeros([1, h_size], dtype=tf.float32), tf.zeros([1, h_size], dtype=tf.float32))  # Initialize RNN state

        for _ in range(num_evaluation):
            while True:
                if terminal:
                    start_time = time.time()
                    airsim_wrapper.reset(point)
                    episode_reward_sum = 0
                    reward_list = []
                    qvalues_list = []
                    action_list = []
                    frame_episode = 0
                    terminal = False

                # Step action
                action, state1 = agent.get_action(0, airsim_wrapper.state, state_in, eval=True)
                _, reward, terminal = airsim_wrapper.step(action)
                frame_episode += 1
                episode_reward_sum += reward
                reward_list.append(reward)
                qvalues_list.append(agent.value(airsim_wrapper.state, state_in))
                action_list.append(action)

                state_in = state1

                if frame_episode == 2000:
                    terminal = True

                # On game-over
                if terminal:
                    df = df.append(
                        {"point": point, 'reward': episode_reward_sum, 'time (s)': time.time() - start_time,
                         'frame': frame_episode, "list of rewards": reward_list, "list of qvalues": qvalues_list,
                         "action": action_list},
                        ignore_index=True)
                    print({"point": point, 'reward': episode_reward_sum, 'time (s)': time.time() - start_time,
                           'frame': frame_episode})
                    break
    return df

if __name__ == "__main__":
    if EVALUATION_DURING_TRAINING == False:
        models = ["DRQN_MBE_3"]
        paths = dict()
        paths[models[0]] = ""

        for model in models:
            print("Evaluation ", model)
            df_train = evaluation_agent(paths[model], num_evaluation=30, h_size=512, starting_points=TRAIN_STARTING_POINTS)
            df_train.to_csv(DATA_USER + "/DRL/results_DRL/definitivo/" + model + "_train.csv", sep=";")

            df_test = evaluation_agent(paths[model], num_evaluation=30, h_size=512, starting_points=TEST_STARTING_POINTS)
            df_test.to_csv(DATA_USER + "/DRL/results_DRL/definitivo/" + model + "_test.csv", sep=";")
    else:
        model = "DRQN"
        paths = {}
        paths[0] = ""
        paths[1] = ""
        paths[2] = ""
        paths[3] = ""
        paths[4] = ""
        paths[5] = ""
        paths[6] = ""

        for i in range(7):
            df_test = evaluation_agent(paths[i], num_evaluation=30, h_size=512, starting_points=TEST_STARTING_POINTS)
            df_test.to_csv(DATA_USER + "/DRL/results_DRL/" + model + "_" + str(i) + "_test.csv", sep=";")
