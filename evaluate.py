# import tensorflow as tf
# from agent.agent import Agent
# from functions import *
# import sys

# if len(sys.argv) != 3:
#     print("Usage: python evaluate.py [stock] [model]")
#     exit()

# stock_name, model_name = sys.argv[1], sys.argv[2]
# model = tf.keras.models.load_model("models/" + model_name)
# window_size = model.input_shape[1]

# agent = Agent(window_size, True, model_name)
# data = getStockDataVec(stock_name)
# l = len(data) - 1
# batch_size = 32

# state = getState(data, 0, int(window_size) + 1)
# total_profit = 0
# agent.inventory = []

# for t in range(l):
#     action = agent.act(state)

#     # sit
#     next_state = getState(data, t + 1, window_size + 1)
#     reward = 0

#     if action == 1:  # buy
#         agent.inventory.append(data[t])
#         print("Buy: " + formatPrice(data[t]))

#     elif action == 2 and len(agent.inventory) > 0:  # sell
#         bought_price = agent.inventory.pop(0)
#         reward = max(data[t] - bought_price, 0)
#         total_profit += data[t] - bought_price
#         print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

#     done = True if t == l - 1 else False
#     agent.memory.append((state, action, reward, next_state, done))
#     state = next_state

#     if done:
#         print("--------------------------------")
#         print(stock_name + " Total Profit: " + formatPrice(total_profit))
#         print("--------------------------------")
from keras.models import load_model
from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 3:
    print("Usage: python evaluate.py [stock] [model]")
    exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("models/" + model_name, compile=False)
window_size = model.input_shape[1]

agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []
sell_count = 0
numb_of_prof = 0
for t in range(l):
    # print("state: ")
    # print(state)
    action = agent.act(state)
    next_state = getState(data, t + 1, window_size + 1)
    reward = 0
    
    if action == 1:  # buy
        agent.inventory.append(data[t])
        print("Buy: " + formatPrice(data[t]))

    elif action == 2 and len(agent.inventory) > 0:  # sell
        sell_count += 1
        bought_price = agent.inventory.pop(0)
        reward = max(data[t] - bought_price, 0)
        total_profit += data[t] - bought_price
        if data[t] - bought_price > 0:
            numb_of_prof += 1
        print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

    done = True if t == l - 1 else False
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state

    if done:

        print("--------------------------------")
        print(stock_name + " Total Profit: " + formatPrice(total_profit))
        print("--------------------------------")
        print(sell_count)
        print(numb_of_prof)
