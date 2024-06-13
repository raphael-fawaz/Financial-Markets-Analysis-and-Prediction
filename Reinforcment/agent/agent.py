import tensorflow as tf
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque
from keras.losses import MeanSquaredError

def custom_mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

def load_custom_model(model_name):
    return load_model("models/" + model_name, custom_objects={"mse": custom_mse})

class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size # normalized previous days
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

		self.model = load_custom_model( model_name) if is_eval else self._model()

	def _model(self):

		model = Sequential()
		model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(units=8, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		optimizer = Adam(learning_rate=0.001)
		model.compile(loss="mse", optimizer=optimizer)

		return model

	def act(self, state):
		if not self.is_eval and random.random() <= self.epsilon:
			return random.randrange(self.action_size)

		options = self.model.predict(state)
		# print("options:")
		# print(options)
		return np.argmax(options[0])

	# def expReplay(self, batch_size):
	# 	mini_batch = []
	# 	l = len(self.memory)
	# 	for i in range(l - batch_size + 1, l):
	# 		mini_batch.append(self.memory[i])

	# 	for state, action, reward, next_state, done in mini_batch:
	# 		target = reward
	# 		if not done:
	# 			target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

	# 		target_f = self.model.predict(state)
	# 		target_f[0][action] = target
	# 		self.model.fit(state, target_f, epochs=1, verbose=0)

	# 	if self.epsilon > self.epsilon_min:
	# 		self.epsilon *= self.epsilon_decay 
	def expReplay(self, batch_size):
		mini_batch = random.sample(self.memory, min(len(self.memory), batch_size))
		# x = np.array(mini_batch)
		# print(mini_batch)
		x =  np.array([experience[0] for experience in mini_batch])
		states = np.array([experience[0] for experience in mini_batch]).reshape(batch_size, self.state_size)
		print(states.shape)

		next_states = np.array([experience[3] for experience in mini_batch]).reshape(batch_size, self.state_size)
		
		target_f = self.model.predict(states)
		target_next = self.model.predict(next_states)
		
		for i, (state, action, reward, next_state, done) in enumerate(mini_batch):
			target = reward if done else reward + self.gamma * np.amax(target_next[i])
			target_f[i][action] = target

		self.model.fit(states, target_f, epochs=1, verbose=0)
		
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
