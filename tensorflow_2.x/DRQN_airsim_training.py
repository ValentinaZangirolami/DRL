import numpy as np
import os
import time
import random
import tensorflow as tf
from scipy import stats

from AirsimEnv.DRQN_classes import ReplayMemory, Agent, AirSimWrapper, QNetwork
from AirsimEnv.DRQN_classes import (BATCH_SIZE, DISCOUNT_FACTOR, FRAMES_BETWEEN_EVAL, TRACE_LENGTH, INPUT_SHAPE,
                           LOAD_REPLAY_MEMORY, EPSILON_ANNELING_FRAMES, MEM_SIZE, NUM_ACTIONS,
                           MIN_REPLAY_MEMORY_SIZE, MAX_EPISODE_LENGTH, WRITE_TENSORBOARD,
                           TOTAL_FRAMES, STARTING_POINTS, EPS_CONST, VDBE, MIN_FRAME_START_TEST, MIN_FRAME_START_SAVE, EPS_INITIAL, NUM_STATES_UPDATE, NUM_ERROR_MASK)

#import rootpath

#def conf_dir(env_key, default_value):
    #p = os.path.expanduser(os.getenv(env_key, default_value))
    #return rootpath.detect(__file__, "^.git$")+p[1:] if p.startswith("./") else p


#DATA_HOME = conf_dir('PC_DATA_HOME', "./data/ext/home")
#DATA_HOST = conf_dir('PC_DATA_HOST', "./data/ext/host")
#DATA_USER = conf_dir('PC_DATA_USER', "./data/ext/user")
#DATA_DESK = conf_dir('PC_DATA_DESK', "~/Desktop")

IP = "127.0.0.1"
PORT = 41451
TYPE_NETWORK = "DRQN_MBE_3"

LOAD_FROM = None
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

SAVE_PATH ="C:/Users/valen/Desktop/magistrale/tesi/csp-drive-rl-master/DRQN/"
TENSORBOARD_DIR = SAVE_PATH + "tensorboard/"

#SAVE_PATH = DATA_USER + "/DRL/" + TYPE_NETWORK + "/"
#TENSORBOARD_DIR = SAVE_PATH + "tensorboard/"

h_size = 512
tau = 0.001


if __name__ == "__main__":

    print(TENSORBOARD_DIR)

    # Initialize TensorBoard writer
    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    # Initialize environment and Q-networks
    airsim_wrapper = AirSimWrapper(ip=IP, port=PORT, input_shape=INPUT_SHAPE)
    rnn_cell = tf.keras.layers.LSTMCell(units=h_size)
    rnn_cell_T = tf.keras.layers.LSTMCell(units=h_size)
    mainQN = QNetwork(h_size, rnn_cell, num_action=NUM_ACTIONS, num_error_mask=NUM_ERROR_MASK,
                      num_states_update=NUM_STATES_UPDATE,batch_size_training=BATCH_SIZE)
    targetQN = QNetwork(h_size, rnn_cell_T, num_action=NUM_ACTIONS, num_error_mask=NUM_ERROR_MASK,
                      num_states_update=NUM_STATES_UPDATE, batch_size_training=BATCH_SIZE)

    # Initialize the replay memory and agent
    replay_memory = ReplayMemory(buffer_size=MEM_SIZE, input_shape=INPUT_SHAPE)
    agent = Agent(mainQN, targetQN, replay_memory, num_actions=NUM_ACTIONS, input_shape=INPUT_SHAPE,
                  batch_size=BATCH_SIZE, eps_annealing_frames=EPSILON_ANNELING_FRAMES, trace_length=TRACE_LENGTH,
                  max_frames=TOTAL_FRAMES)

    # Load or initialize variables
    if LOAD_FROM is None:
        frame_number = 0
        rewards = []
        loss_list = []
        action_list = []
        eval_list = []
        agent.list_vdbe.append(EPS_INITIAL)
    else:
        print('Loading from', LOAD_FROM)
        action_list = list(np.load(SAVE_PATH + '/action.npy', allow_pickle=True))
        meta = agent.load(LOAD_FROM, VDBE, LOAD_REPLAY_MEMORY)
        frame_number = meta['frame_number']
        eval_list = list(
            np.load(SAVE_PATH + '/evaluation.npy', allow_pickle=True)) if frame_number > MIN_FRAME_START_TEST else []
        rewards = meta['rewards']
        loss_list = meta['loss_list']
        #agent.main_drqn.load_weights(os.path.join(LOAD_FROM, 'main_drqn_model'))
        #agent.target_drqn.load_weights(os.path.join(LOAD_FROM, 'target_drqn_model')

    initial_start_time = time.time()
    try:
        # Training loop
        episode_number = 1
        while frame_number < TOTAL_FRAMES:
            # Training
            state_in = (tf.zeros([1, h_size], dtype=tf.float32), tf.zeros([1, h_size], dtype=tf.float32))
            epoch_frame = 0
            start_time_progress = time.time()

            while epoch_frame < FRAMES_BETWEEN_EVAL:
                airsim_wrapper.reset(random.choice(STARTING_POINTS))
                episode_reward_sum = 0
                state_buffer, action_buffer, next_state_buffer, reward_buffer, terminal_buffer = [], [], [], [], []


                for j in range(MAX_EPISODE_LENGTH):
                    frame_time = time.time()
                    # Get action
                    frame = airsim_wrapper.state
                    action, state1 = agent.get_action(frame_number, frame, state_in, eval=False,
                                                          eps_const=EPS_CONST, vdbe=VDBE)
                    action_list.append(action)

                        # Take step
                    next_frame, reward, terminal = airsim_wrapper.step(action)
                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward

                    state_in = state1

                    if frame.shape != INPUT_SHAPE or next_frame.shape != INPUT_SHAPE:
                        print("Dimension of frame is wrong!")
                        break

                    # Add experience to buffer
                    state_buffer.append(frame)
                    next_state_buffer.append(next_frame)
                    action_buffer.append(action)
                    reward_buffer.append(reward)
                    terminal_buffer.append(terminal)

                    # Update agent
                    if frame_number % 4 == 0 and len(agent.replay_memory.action) > MIN_REPLAY_MEMORY_SIZE:
                        state_train = (tf.zeros([BATCH_SIZE, h_size], dtype=tf.float32), tf.zeros([BATCH_SIZE, h_size], dtype=tf.float32))
                        agent.update_target_network(tau)
                        loss = agent.learn(batch_size=BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                                               frame_number=frame_number, trace_length=TRACE_LENGTH,
                                               state_train=state_train)
                        loss_list.append(loss)

                    # Break the loop when the game is over
                    if terminal:
                        break

                rewards.append(episode_reward_sum)
                episode_number += 1

                # Add episode to replay memory
                if len(state_buffer) >= TRACE_LENGTH:
                    # Convert buffers to numpy arrays and ensure correct dtype for states
                    state_buffer_np = np.array(state_buffer, dtype=np.uint8)
                    next_state_buffer_np = np.array(next_state_buffer, dtype=np.uint8)
                    action_buffer_np = np.array(action_buffer)
                    reward_buffer_np = np.array(reward_buffer)
                    terminal_buffer_np = np.array(terminal_buffer)

                    # Add experience to replay memory
                    agent.add_experience(state_buffer_np, action_buffer_np, reward_buffer_np, next_state_buffer_np,
                                             terminal_buffer_np)

                # Output the progress and write to TensorBoard every 100 episodes
                if len(rewards) % 100 == 0:
                    avg_reward = np.mean(rewards[-100:])
                    avg_loss = np.mean(loss_list[-100:]) if len(loss_list) > 0 else 0

                    # Write summaries to TensorBoard
                    if WRITE_TENSORBOARD==True:
                        with writer.as_default():
                            tf.summary.scalar('Average Reward', avg_reward, step=frame_number)
                            tf.summary.scalar('Average Loss', avg_loss, step=frame_number)
                        writer.flush()

                # Output the progress every 100 games
                if len(rewards) % 100 == 0:
                    print(f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  '
                              f'Average reward: {np.mean(rewards[-100:]):0.1f}  Time taken: {time.time() - start_time_progress:.1f} s')
                    start_time_progress = time.time()

                # Save model
                if len(rewards) % 500 == 0 and frame_number > MIN_FRAME_START_SAVE and SAVE_PATH is not None:
                    agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', vdbe=VDBE,
                               frame_number=frame_number, rewards=rewards, loss_list=loss_list)
                    np.save(SAVE_PATH + '/action.npy', action_list)

            # Evaluation every `FRAMES_BETWEEN_EVAL` frames
            if frame_number > MIN_FRAME_START_TEST:
                eval_rewards = []
                evaluate_frame_number = 0
                frame_episode = 0

                terminal = True
                for point in STARTING_POINTS:

                    state_in = (tf.zeros([1, h_size], dtype=tf.float32), tf.zeros([1, h_size], dtype=tf.float32))
                    while True:
                        if terminal:
                            airsim_wrapper.reset(point)
                            episode_reward_sum = 0
                            frame_episode = 0
                            terminal = False

                        # Step action
                        action, state1 = agent.get_action(frame_number, airsim_wrapper.state, state_in,  eval=True)
                        _, reward, terminal = airsim_wrapper.step(action)
                        evaluate_frame_number += 1
                        frame_episode += 1
                        episode_reward_sum += reward
                        state_in = state1

                        # On game-over
                        if terminal:
                            print("Reward per episode: ", episode_reward_sum)
                            eval_rewards.append(episode_reward_sum)
                            break

                if len(eval_rewards) > 0:
                    final_score = np.mean(eval_rewards)
                else:
                    # In case the game is longer than the number of frames allowed
                    final_score = episode_reward_sum

                # Print score and write to tensorboard
                print('Evaluation score:', final_score)
                eval_list.append(final_score)
                np.save(SAVE_PATH + '/evaluation.npy', eval_list)
                if WRITE_TENSORBOARD==True:
                    with writer.as_default():
                        tf.summary.scalar('Evaluation Score', final_score, step=frame_number)
                    writer.flush()



    except KeyboardInterrupt:
        print('\nTraining exited early.')
        if SAVE_PATH is not None:
            print('Saving...')
            agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', vdbe=VDBE, frame_number=frame_number,
                           rewards=rewards, loss_list=loss_list)
            np.save(SAVE_PATH + '/action.npy', action_list)
            print('Saved.')