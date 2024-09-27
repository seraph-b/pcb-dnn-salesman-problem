from env.package.pcb_env import PCBEnv
from utils.package.utils import *

from os.path import isfile
from collections import deque
from random import sample
from random import random
from tensorflow import keras as K
import tensorflow as tf
import numpy as np

class Model():
    def __init__(self,input_shape,output_shape=None,action_shape=None):
        super(Model,self).__init__()
        self.actions = [x for x in range(4)]
        self.action_shape = action_shape
        self.model = None

    def create_critic(self):
        """

        Creates a Keras model with two inputs of shapes (10000,) and
        (4,) respectively. Two hidden layers precede a dropout layer
        which then follows into another hidden and merge layer. The
        model is saved as self.model.

        Inspired by Patel (2020).

        """
        weight_init = 'truncated_normal'
        bias_init = 'random_normal'
        self.inl1 = K.layers.Input(shape=(10000,),name='state_input')
        self.inl2 = K.layers.Input(shape=(4,),name='action_input')
        d1 = K.layers.Dense(units=4096,activation='relu',name='hidden1',kernel_initializer=weight_init,bias_initializer=bias_init)(self.inl1)
        d1 = K.layers.Dense(units=4096,activation='relu',name='hidden1_1',kernel_initializer=weight_init,bias_initializer=bias_init)(d1)
        self.dropout = K.layers.Dropout(0.2)(d1)
        d2_1 = K.layers.Dense(units=1024,activation='relu',name='hidden2',kernel_initializer=weight_init,bias_initializer=bias_init)(self.dropout)
        d2_2 = K.layers.Dense(units=1024,activation='relu',name='hidden_action',kernel_initializer=weight_init,bias_initializer=bias_init)(self.inl2)
        merge = K.layers.Add()([d2_1,d2_2])
        d3 = K.layers.Dense(units=128,activation='relu',name='hidden3',kernel_initializer=weight_init,bias_initializer=bias_init)(merge)
        d4 = K.layers.Dense(units=32,activation='relu',name='hidden4',kernel_initializer=weight_init,bias_initializer=bias_init)(d3)
        self.output_layer = K.layers.Dense(units=1,activation='relu',name='output',kernel_initializer=weight_init,bias_initializer=bias_init)(d4)
        self.model = K.Model(inputs=[self.inl1,self.inl2],outputs=self.output_layer)
        self.model.compile(loss='mse',optimizer=K.optimizers.Adam())

    def create_actor(self):
        """

        Creates a Keras model with an input of shape (10000,). Two
        hidden layers precede a dropout layer which then follows
        into two more hidden layers. The model is saved as
        self.model.

        """
        weight_init = 'truncated_normal'
        bias_init = 'random_normal'
        model = K.models.Sequential()
        self.inl = K.layers.Input(shape=(10000,),dtype='float32')
        model.add(self.inl)
        model.add(K.layers.Dense(units=4096,activation='relu',kernel_initializer=weight_init,bias_initializer=bias_init))
        model.add(K.layers.Dense(units=4096,activation='relu',kernel_initializer=weight_init,bias_initializer=bias_init))
        self.dropout = K.layers.Dropout(0.2)
        model.add(self.dropout)
        model.add(K.layers.Dense(units=1024,activation='relu',kernel_initializer=weight_init,bias_initializer=bias_init))
        model.add(K.layers.Dense(units=128,activation='relu',kernel_initializer=weight_init,bias_initializer=bias_init))
        model.add(K.layers.Dense(units=self.action_shape,activation='softmax',kernel_initializer=weight_init,bias_initializer=bias_init))
        model.compile(optimizer=K.optimizers.Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
        self.model = model

    def create_optimizer(self):
        """

        Creates an Adam optimizer and stores it as
        self.optimizer. This is composed of the weights of
        the actor model and gradients calculated using
        tf.gradients().

        Draws from Patel (2020).

        """
        self.critic_gradient = tf.compat.v1.placeholder(dtype=tf.float32,shape=(1,4))
        self.weights = self.model.trainable_weights
        self.actor_gradient = tf.gradients(ys=self.model.output,xs=self.weights,grad_ys=-self.critic_gradient)
        self.optimizer = tf.compat.v1.train.AdamOptimizer().apply_gradients(zip(self.actor_gradient,self.weights))

    def create_action_gradient(self):
        """

        Calculates the gradient of the critic model and
        stores this as self.critic_gradient.

        """
        self.critic_gradient = tf.gradients(ys=self.output_layer,xs=self.inl2)

    def predict(self,state):
        """

        Uses the model to predict the next action using the
        state as the network's input.

        :param state: The current state of the environment.
        :return: Integer from 0-3 representing the next
        action a two-dimensional array of probabilities.

        """
        probs = self.model.predict(state)[0]
        if random() < 0.1:
            return np.random.choice(self.actions,p=probs),probs
        action = 0
        for i in range(len(probs)):
            if probs[i] > probs[action]:
                action = i
            elif probs[i] == probs[action] and random() > 0.5:
                action = i
        return action,probs

    def predict_reward(self,state,action):
        """

        Uses the model to predict the next reward using the
        state and action probabilities as the network's input.

        :param state: The current state of the environment.
        :param action: An array of probabilities associated
        with taking the next action in state.
        :return: The expected reward as a [[float32]].

        """
        expected_reward = self.model.predict({'state_input':state,'action_input':action})
        return expected_reward

def train_actor(actor,critic,state,session):
    """

    Uses the predifened optimizer and critic's gradient
    to train the actor.

    :param actor: The actor model.
    :param critic: The critic model.
    :param state: The current state of the environment.
    :param session: The current tf.compat.v1.Session().

    Draws from Patel (2020).

    """
    gradient = session.run(critic.critic_gradient,feed_dict={critic.inl1:state,critic.inl2:actor.predict(state)[1].reshape((1,4))})[0]
    session.run(actor.optimizer,feed_dict={actor.inl:state,actor.critic_gradient:gradient})

def train_critic(model,state,action,reward):
    """

    Trains the critic by fitting the model with the current
    state and action probabilities as input and total reward
    as the desired output.

    :param model: The critic model.
    :param state: The current state of the environment.
    :param action: An array of probabilities associated
    with taking the next action in state.
    :param reward: The total expected reward for this
    state-action pair.

    """
    model.model.fit(x=[state,action],
        y=np.array([[reward]]),batch_size=1,verbose=False)

def train(actor,critic,history,batch_size,session):
    """

    Pulls a batch from the history and trains both models
    over this data.

    :param actor: The actor model.
    :param critic: The critic model.
    :param history: A deque containing all state-action pairs.
    :param batch_size: An integer determining the number of
    samples to train over.
    :param session: The current tf.compat.v1.Session().

    """
    if batch_size > len(history):
        batch_size = len(history)

    critic.dropout.training = True
    batch = sample(history,batch_size)
    for t in batch:
        state,action,next_state,reward,done = t
        if not done:
            # Add the discounted predicted reward if state is not terminal
            reward += 0.95*critic.predict_reward(next_state,actor.predict(next_state)[1].reshape((1,4)))[0][0]
        train_critic(critic,state,action,reward)
    critic.dropout.training = False
    
    actor.dropout.training = True
    batch = sample(history,batch_size)
    for t in batch:
        state,_,_,_,_ = t
        train_actor(actor,critic,state,session)
    actor.dropout.training = False

if __name__ == '__main__':
    # Initialize variables
    env = PCBEnv()
    cfg = Config()
    actor_file = './training/acp.ckpt'
    critic_file = './training/ccp.ckpt'
    cur_episode = 0
    reward_acc = 0

    # Create Tensorflow device
    tf.compat.v2.enable_v2_behavior()
    tf.compat.v1.disable_eager_execution()
    session = tf.compat.v1.Session()

    # Create history deque
    history = deque(maxlen=4000)

    # Import configuration file
    if isfile('config.yaml'):
        cfg = to_cfg(read_yaml('config.yaml'))
    else:
        write_yaml(d=to_dict(cfg))

    # Create models
    actor = Model(input_shape=env.observation_space.shape,action_shape=4)
    actor.create_actor()
    actor.create_optimizer()

    critic = Model(input_shape=env.observation_space.shape,action_shape=4)
    critic.create_critic()
    critic.create_action_gradient()

    session.run(tf.compat.v1.global_variables_initializer())

    # Begin learning cycle
    state = scale(env.reset().reshape((1, 10000)).astype('float32'),5)
    while True:
        # Display a window with a render of the environment
        # env.render()
        
        # Predict and perform the next action
        action,probs = actor.predict(state)
        next_state,reward,done,_ = env.step(action)
        
        # Scale and shape input
        reward_acc += reward
        probs = probs.reshape((1,4))
        #reward = scale(reward,100)
        next_state = scale(next_state.reshape((1, 10000)).astype('float32'),5)

        # Save last interaction to history
        history.append([state,probs,next_state,reward,done])

        # Train with sample data and set new state
        state = next_state

        # Train models and reset the environment if in terminal state
        if done:
            train(actor,critic,history,50,session)
            state = scale(env.reset().reshape((1, 10000)).astype('float32'),5)

            if cur_episode % 100 == 0:
                actor.model.save_weights(actor_file)
                critic.model.save_weights(critic_file)

            print('Total reward in episode '+str(cur_episode)+': '+str(reward_acc))
            cur_episode += 1
            reward_acc = 0

"""
References:
  Patel, Y., 2020. Reinforcement Learning W/ Keras + Openai: Actor-Critic Models. [online] Medium.
  Available at: <https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69>.
"""