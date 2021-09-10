import numpy as np
import os
import time
import random
import tensorflow as tf

from scipy import stats

from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract, LSTM, TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling
from AirsimEnv.DRQN_bayes_classes import ReplayMemory, Agent, AirSimWrapper
from AirsimEnv.DRQN_bayes_classes import (BATCH_SIZE, DISCOUNT_FACTOR, FRAMES_BETWEEN_EVAL, TRACE_LENGTH, INPUT_SHAPE,
                           LOAD_REPLAY_MEMORY, MEM_SIZE, NUM_ACTIONS,
                           MIN_REPLAY_MEMORY_SIZE, MAX_EPISODE_LENGTH,
                           TOTAL_FRAMES, UPDATE_FREQ, WRITE_TENSORBOARD, STARTING_POINTS, WEIGHT_PATH)


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

IP = "127.0.0.1"
PORT = 41451
TYPE_NETWORK = "DRQN_bayes"
TL = False

LOAD_FROM = 'C:/Users/valen/Desktop/magistrale/tesi/csp-drive-rl-master/DRQN_bayes/save-00439531'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

if TL:
    SAVE_PATH = "C:/Users/valen/Desktop/magistrale/tesi/csp-drive-rl-master/" + TYPE_NETWORK + "_TL/"
    TENSORBOARD_DIR = SAVE_PATH + "tensorboard/"
else:
    SAVE_PATH = "C:/Users/valen/Desktop/magistrale/tesi/csp-drive-rl-master/" + TYPE_NETWORK + "/"
    TENSORBOARD_DIR = SAVE_PATH + "tensorboard/"


def build_deep_network(num_actions, input_shape, transfer_learning=TL):

    model_input = Input(shape=input_shape)
    x = Lambda(lambda img: img / 255)(model_input)  # normalize by 255
    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False, name="conv_1")(x)
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False, name="conv_2")(x)
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False, name="conv_3")(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(512, activation='tanh')(x)
    val_stream = Dense(512, name="dense_val_stream")(x)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.), name="dense_val")(val_stream)
    adv_stream = Dense(512, name="dense_adv_action")(x)
    adv = Dense(num_actions, kernel_initializer=VarianceScaling(scale=2.), name="dense_adv")(adv_stream)

    # Combine streams into Q-Values
    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean
    q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])
    model = Model(model_input, q_vals)

    model.compile(Adam(), loss=tf.keras.losses.MeanSquaredError())
    print(model.summary())

    if transfer_learning:
        model.load_weights(WEIGHT_PATH, by_name=True)
        print("Loading weights")
    else:
        print("Not loading weights")

    return model


def dbs(h, fun):
    incrementi = []
    for i in range(len(fun)):
        if i+h+1 <= len(fun):
            incrementi.append(fun[h+i] - fun[i])
    return incrementi


def wdc(dbs_list):
    incr_n = 0
    incr_p = 0
    for i in range(len(dbs_list)):
        if dbs_list[i] < 0:
            incr_n += dbs_list[i]
        else:
            incr_p += dbs_list[i]
    return [incr_n, incr_p]

if __name__ == "__main__":
    print(TENSORBOARD_DIR)

    airsim_wrapper = AirSimWrapper(ip=IP, port=PORT, input_shape=INPUT_SHAPE)
    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    MAIN_DQN = build_deep_network(num_actions=NUM_ACTIONS, input_shape=INPUT_SHAPE)
    TARGET_DQN = build_deep_network(num_actions=NUM_ACTIONS, input_shape=INPUT_SHAPE)

    replay_memory = ReplayMemory(buffer_size=MEM_SIZE)
    agent = Agent(MAIN_DQN, TARGET_DQN, replay_memory, num_actions=NUM_ACTIONS, input_shape=INPUT_SHAPE,
                  batch_size=BATCH_SIZE, trace_length=TRACE_LENGTH)

    if LOAD_FROM is None:
        frame_number = 0
        rewards = []
        loss_list = []
    else:
        print('Loading from', LOAD_FROM)
        meta = agent.load(LOAD_FROM, LOAD_REPLAY_MEMORY)

        # Apply information loaded from meta
        frame_number = meta['frame_number']
        rewards = meta['rewards']
        loss_list = meta['loss_list']

    initial_start_time = time.time()
    action_episode = []
    number_episode = []
    time_collision = []
    epsilon_li = []
    try:
        with writer.as_default():
            while frame_number < TOTAL_FRAMES:

                # Training
                epoch_frame = 0
                start_time_progress = time.time()

                episode_n = 1
                while epoch_frame < FRAMES_BETWEEN_EVAL:

                    point_start = random.choice(STARTING_POINTS)
                    airsim_wrapper.reset(point_start)
                    start_time_episode = time.time()
                    episode_buffer = []

                    episode_reward_sum = 0
                    j = 0

                    for j in range(MAX_EPISODE_LENGTH):

                        j += 1
                        number_episode.append(episode_n)

                        frame_time = time.time()
                        # get action
                        frame = airsim_wrapper.state
                        action = agent.get_action(frame_number, frame, eval=False)
                        action_episode.append(action)


                        # take step
                        next_frame, reward, terminal = airsim_wrapper.step(action)
                        frame_number += 1
                        epoch_frame += 1
                        episode_reward_sum += reward
                        epsilon_li.append(agent.alpha/(agent.alpha+agent.beta))

                        # compute model means for exploration
                        G_Q = reward + DISCOUNT_FACTOR * np.amax(agent.value(next_frame))
                        G_U = reward + DISCOUNT_FACTOR * np.mean(agent.value(next_frame))

                        # update epsilon
                        agent.update_posterior(data=(G_Q, G_U))

                        # add experience to replay memory
                        if frame.shape != INPUT_SHAPE or next_frame.shape != INPUT_SHAPE:
                            print("Dimension of frame is wrong!")
                        else:
                            state_frame = np.reshape(frame, (1, 66, 200, 3))
                            nextstate_frame = np.reshape(next_frame, (1, 66, 200, 3))
                            episode_buffer.append(np.reshape(np.array([np.asarray(state_frame, dtype=np.uint8), action, reward, np.asarray(nextstate_frame, dtype=np.uint8), terminal]), [1,5]))

                        # update agent
                        if frame_number % 4 == 0 and len(agent.replay_memory.buffer) + 1 > MIN_REPLAY_MEMORY_SIZE:
                            loss, _ = agent.learn(batch_size=BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                                                  frame_number=frame_number, trace_length=TRACE_LENGTH)
                            loss_list.append(loss)

                        elif frame_number % 4 == 0:
                            time.sleep(0.10)

                        # Update target network
                        if frame_number % UPDATE_FREQ == 0 and frame_number > MIN_REPLAY_MEMORY_SIZE:
                            agent.update_target_network()

                        # Break the loop when the game is over
                        if terminal:
                            terminal = False
                            episode_n += 1
                            minutes_collision = divmod(time.time() - start_time_episode, 60)
                            time_collision.append([minutes_collision, episode_n]) # try TTC (not same in VM)
                            break

                        #print("Time of frame evaluation:", time.time() - frame_time)
                    if j >= TRACE_LENGTH:
                        agent.add_experience(list(zip(np.array(episode_buffer))))
                    rewards.append(episode_reward_sum)

                    # Output the progress every 100 games
                    if len(rewards) % 100 == 0:
                        # Write to TensorBoard
                        if WRITE_TENSORBOARD:
                            tf.summary.scalar('Reward', np.mean(rewards[-100:]), frame_number)
                            tf.summary.scalar('Loss', np.mean(loss_list[-100:]), frame_number)
                            writer.flush()

                        hours = divmod(time.time() - initial_start_time, 3600)
                        minutes = divmod(hours[1], 60)
                        minutes_100 = divmod(time.time() - start_time_progress, 60)
                        wdc_list = wdc(dbs(1, rewards[-100:]))
                        print(f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  '
                              f'Average reward: {np.mean(rewards[-100:]):0.1f}  Time taken: {(minutes_100[0]):.1f}  '
                              f'Total time taken: {(int(hours[0]))}:{(int(minutes[0]))}:{(minutes[1]):0.1f} '
                              f'Dev. Standard reward: {np.std(rewards[-100:]):0.1f} IQR: {stats.iqr(rewards[-100:]):0.1f}  '
                              f'Min: {min(rewards[-100:]):0.1f}  Max: {max(rewards[-100:]):0.1f} '
                              f'WDC_n: {wdc_list[0]:0.1f}  WDC_p: {wdc_list[1]:0.1f} ')
                        start_time_progress = time.time()

                    # Save model
                    if len(rewards) % 500 == 0 and SAVE_PATH is not None:
                        agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number,
                                   rewards=rewards, loss_list=loss_list)
                # save action
                np.savez(SAVE_PATH + 'action_episode', x=action_episode, y=number_episode, z=epsilon_li)
                # Evaluation every `FRAMES_BETWEEN_EVAL` frames
                eval_rewards = []
                evaluate_frame_number = 0

                terminal = True
                for point in STARTING_POINTS:
                    while True:
                        if terminal:
                            airsim_wrapper.reset(point)
                            episode_reward_sum = 0
                            frame_episode = 0
                            terminal = False

                        # Step action
                        action = agent.get_action(frame_number, airsim_wrapper.state, eval=True)
                        _, reward, terminal = airsim_wrapper.step(action)
                        evaluate_frame_number += 1
                        frame_episode += 1
                        episode_reward_sum += reward

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
                if WRITE_TENSORBOARD:
                    tf.summary.scalar('Evaluation score', final_score, frame_number)
                    writer.flush()

            agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number,
                       rewards=rewards, loss_list=loss_list)

    except KeyboardInterrupt:
        print('\nTraining exited early.')
        writer.close()

        if SAVE_PATH is None:
            try:
                SAVE_PATH = input(
                    'Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with ctrl+c. ')
            except KeyboardInterrupt:
                print('\nExiting...')

        if SAVE_PATH is not None:
            print('Saving...')
            agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards,
                       loss_list=loss_list)
            print('Saved.')