import numpy as np
import os
import time
import random
import tensorflow as tf
from scipy import stats

from AirsimEnv.bayesian import Beta, Average
from AirsimEnv.DRQN_classes import ReplayMemory, Agent, AirSimWrapper, Qnetwork
from AirsimEnv.DRQN_classes import (BATCH_SIZE, DISCOUNT_FACTOR, FRAMES_BETWEEN_EVAL, TRACE_LENGTH, INPUT_SHAPE,
                           LOAD_REPLAY_MEMORY, EPSILON_ANNELING_FRAMES, MEM_SIZE, NUM_ACTIONS,
                           MIN_REPLAY_MEMORY_SIZE, MAX_EPISODE_LENGTH, ALPHA, BETA,
                           TOTAL_FRAMES, EPS_DESC_EPISODE, STARTING_POINTS, BAYES, EPS_CONST, VDBE, EPS_DESC, SOFTMAX, EPS_GREEDY, MBE, MIN_FRAME_START_TEST, MIN_FRAME_START_SAVE, EPS_INITIAL, NUM_STATES_UPDATE, NUM_ERROR_MASK)

import rootpath

def conf_dir(env_key, default_value):
    p = os.path.expanduser(os.getenv(env_key, default_value))
    return rootpath.detect(__file__, "^.git$")+p[1:] if p.startswith("./") else p


DATA_HOME = conf_dir('PC_DATA_HOME', "./data/ext/home")
DATA_HOST = conf_dir('PC_DATA_HOST', "./data/ext/host")
DATA_USER = conf_dir('PC_DATA_USER', "./data/ext/user")
DATA_DESK = conf_dir('PC_DATA_DESK', "~/Desktop")


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)

tf.compat.v1.disable_eager_execution()

IP = "127.0.0.1"
PORT = 41451
TYPE_NETWORK = "DRQN_MBE_3"
TL = False

LOAD_FROM = None
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)
tf.compat.v1.random.set_random_seed(123)
tf.compat.v1.set_random_seed(123)

SAVE_PATH = DATA_USER + "/DRL/" + TYPE_NETWORK + "/"
TENSORBOARD_DIR = SAVE_PATH + "tensorboard/"

h_size = 512
tau = 0.001


# Update Target Network: tau is a parameter that allow us to update TN with tau% weights of Main network and (1-tau)% weights of TN
# (Credit to Juliani A. for this and the structure of Recurrent CNN)
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
    total_vars = len(tf.compat.v1.trainable_variables())
    a = tf.compat.v1.trainable_variables()[0].eval(session=sess)
    b = tf.compat.v1.trainable_variables()[total_vars//2].eval(session=sess)
    if a.all() != b.all():
        print("Target Set Failed")


if __name__ == "__main__":

    print(TENSORBOARD_DIR)

    airsim_wrapper = AirSimWrapper(ip=IP, port=PORT, input_shape=INPUT_SHAPE)
    tf.compat.v1.reset_default_graph()
    # We define the cells for the primary and target q-networks
    cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    cellT = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    mainQN = Qnetwork(h_size, cell, 'main', num_action=NUM_ACTIONS, num_error_mask=NUM_ERROR_MASK, num_states_update=NUM_STATES_UPDATE)
    targetQN = Qnetwork(h_size, cellT, 'target', num_action=NUM_ACTIONS, num_error_mask=NUM_ERROR_MASK, num_states_update=NUM_STATES_UPDATE)

    beta = Beta(ALPHA, BETA)
    average = Average()

    init = tf.compat.v1.global_variables_initializer()

    saver = tf.compat.v1.train.Saver(max_to_keep=5)

    trainables = tf.compat.v1.trainable_variables()

    targetOps = updateTargetGraph(trainables, tau)

    replay_memory = ReplayMemory(buffer_size=MEM_SIZE, input_shape=INPUT_SHAPE)
    agent = Agent(mainQN, targetQN, replay_memory, beta, average, num_actions=NUM_ACTIONS, input_shape=INPUT_SHAPE,
                  batch_size=BATCH_SIZE, eps_annealing_frames=EPSILON_ANNELING_FRAMES, trace_length=TRACE_LENGTH,
                  max_frames=TOTAL_FRAMES)
    with tf.compat.v1.Session() as session:
        writer = tf.compat.v1.summary.FileWriter(TENSORBOARD_DIR, session.graph)


        if LOAD_FROM is None:
            frame_number = 0
            rewards = []
            loss_list = []
            action_list = []
            eval_list = []
            agent.list_vdbe.append(EPS_INITIAL)
            session.run(init)
        else:
            print('Loading from', LOAD_FROM)
            action_list = list(np.load(SAVE_PATH + '/action.npy', allow_pickle=True))
            meta = agent.load(LOAD_FROM, SOFTMAX, BAYES, VDBE, LOAD_REPLAY_MEMORY)
            # Apply information loaded from meta
            frame_number = meta['frame_number']
            if frame_number > MIN_FRAME_START_TEST:
                eval_list = list(np.load(SAVE_PATH + '/evaluation.npy', allow_pickle=True))
            else:
                eval_list = []
            rewards = meta['rewards']
            loss_list = meta['loss_list']
            ckpt = tf.train.get_checkpoint_state(LOAD_FROM)
            saver.restore(session, ckpt.model_checkpoint_path)

        initial_start_time = time.time()
        try:

            updateTarget(targetOps, session)
            episode_number = 1
            while frame_number < TOTAL_FRAMES:
                # Training
                state_in = (np.zeros([1, h_size]), np.zeros([1, h_size]))
                epoch_frame = 0
                start_time_progress = time.time()

                while epoch_frame < FRAMES_BETWEEN_EVAL:

                    airsim_wrapper.reset(random.choice(STARTING_POINTS))
                    state_buffer = []
                    action_buffer = []
                    next_state_buffer = []
                    reward_buffer = []
                    terminal_buffer = []


                    episode_reward_sum = 0
                    j = 0

                    for j in range(MAX_EPISODE_LENGTH):

                        j+=1

                        frame_time = time.time()
                        # get action
                        frame = airsim_wrapper.state
                        action, state1 = agent.get_action(frame_number, episode_number, frame, state_in, session=session, bayes=BAYES, eps_const=EPS_CONST, vdbe=VDBE, eps_episode=EPS_DESC_EPISODE, softmax=SOFTMAX, eps_greedy=EPS_GREEDY, mbe=MBE)
                        action_list.append(action)


                        # take step
                        next_frame, reward, terminal = airsim_wrapper.step(action)
                        frame_number += 1
                        epoch_frame += 1
                        episode_reward_sum += reward

                        state_in = state1

                        if BAYES == True and len(agent.replay_memory.action) > MIN_REPLAY_MEMORY_SIZE:
                            G_Q = reward + DISCOUNT_FACTOR * np.amax(agent.value(next_frame, state_in, session=session))
                            G_U = reward + DISCOUNT_FACTOR * np.mean(agent.value(next_frame, state_in, session=session))
                            agent.update_posterior(data=(G_Q, G_U))


                        # add experience
                        if frame.shape != INPUT_SHAPE or next_frame.shape != INPUT_SHAPE:
                            print("Dimension of frame is wrong!")
                        else:
                            state_buffer.append(np.array(np.reshape(frame, (66, 200, 3)), dtype=np.uint8))
                            next_state_buffer.append(np.array(np.reshape(next_frame, (66, 200, 3)), dtype=np.uint8))
                            action_buffer.append(action)
                            reward_buffer.append(reward)
                            terminal_buffer.append(terminal)

                        # update agent
                        if frame_number % 4 == 0 and frame_number > 50000 and len(agent.replay_memory.action) > BATCH_SIZE and EPS_DESC==True:
                            state_train = (np.zeros([BATCH_SIZE, h_size]), np.zeros([BATCH_SIZE, h_size]))
                            updateTarget(targetOps, session)
                            loss, _ = agent.learn(batch_size=BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                                                  frame_number=frame_number, trace_length=TRACE_LENGTH, state_train=state_train, session=session)
                            loss_list.append(loss)

                        elif frame_number % 4 == 0 and len(agent.replay_memory.action) > MIN_REPLAY_MEMORY_SIZE and EPS_DESC == False:
                            if VDBE==True:
                                q_old = agent.value(frame, state_in, session=session)[action]
                            state_train = (np.zeros([BATCH_SIZE, h_size]), np.zeros([BATCH_SIZE, h_size]))
                            updateTarget(targetOps, session)
                            loss, _ = agent.learn(batch_size=BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                                                  frame_number=frame_number, trace_length=TRACE_LENGTH,
                                                  state_train=state_train, session=session)
                            loss_list.append(loss)
                            if VDBE==True:
                                q_new = agent.value(frame, state_in, session=session)[action]
                                agent.update_vdbe(q_new-q_old)

                        elif frame_number % 4 == 0:
                            time.sleep(0.10)

                        # Break the loop when the game is over
                        if terminal:
                            terminal = False
                            break

                        #print("Time of frame evaluation:", time.time() - frame_time)

                    rewards.append(episode_reward_sum)
                    episode_number += 1

                    #add episode to replay memory
                    if j >= TRACE_LENGTH:
                        agent.add_experience(np.array(state_buffer), np.array(action_buffer), np.array(reward_buffer), np.array(next_state_buffer), np.array(terminal_buffer))

                    # Output the progress every 100 games
                    if len(rewards) % 100 == 0:

                        hours = divmod(time.time() - initial_start_time, 3600)
                        minutes = divmod(hours[1], 60)
                        minutes_100 = divmod(time.time() - start_time_progress, 60)

                        print(f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  '
                              f'Average reward: {np.mean(rewards[-100:]):0.1f}  Time taken: {(minutes_100[0]):.1f}  '
                              f'Total time taken: {(int(hours[0]))}:{(int(minutes[0]))}:{(minutes[1]):0.1f} '
                              f'Dev. Standard reward: {np.std(rewards[-100:]):0.1f} IQR: {stats.iqr(rewards[-100:]):0.1f} '
                              f'Min: {min(rewards[-100:]):0.1f}  Max: {max(rewards[-100:]):0.1f} ')
                        start_time_progress = time.time()

                    # Save model
                    if len(rewards) % 500 == 0 and frame_number > MIN_FRAME_START_SAVE and SAVE_PATH is not None:
                        agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', SOFTMAX, BAYES, VDBE, frame_number=frame_number,
                                   rewards=rewards, loss_list=loss_list)
                        saver.save(session, f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}' + '/model.cptk')
                        np.save(SAVE_PATH + '/action.npy', action_list)


                # Evaluation every `FRAMES_BETWEEN_EVAL` frames
                if frame_number > MIN_FRAME_START_TEST:
                    eval_rewards = []
                    evaluate_frame_number = 0

                    terminal = True
                    for point in STARTING_POINTS:

                        state_in = (np.zeros([1, h_size]), np.zeros([1, h_size]))
                        while True:
                            if terminal:
                                airsim_wrapper.reset(point)
                                episode_reward_sum = 0
                                frame_episode = 0
                                terminal = False

                        # Step action
                            action, state1, _ = agent.get_action(frame_number, episode_number, airsim_wrapper.state, state_in, session=session, eval=True)
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

            agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', SOFTMAX, BAYES, VDBE,
                       frame_number=frame_number,
                       rewards=rewards, loss_list=loss_list)
            saver.save(session, f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}' + '/model.cptk')
            np.save(SAVE_PATH + '/action.npy', action_list)


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
                agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', SOFTMAX, BAYES, VDBE,
                           frame_number=frame_number,
                           rewards=rewards, loss_list=loss_list)

                saver.save(session, f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}' + '/model.cptk')
                np.save(SAVE_PATH + '/action.npy', action_list)
                print('Saved.')