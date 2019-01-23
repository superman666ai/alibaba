import re
import pandas as pd
import numpy as np
import tensorflow as tf

# 读取csv数据
tb_kline_m5 = pd.read_csv('data/tb_kline_m5.csv', sep=',', encoding='utf-8', lineterminator='\n')

tb_kline_m5 = tb_kline_m5.head(5)

code_list = ['ag', 'al', 'au', 'bu', 'cu', 'fu', 'hc', 'ni', 'pb', 'rb', 'ru', 'sn', 'wr', 'zn', 'sc', 'a', 'b', 'bb',
             'c', 'cs', 'fb', 'i', 'j', 'jd', 'jm', 'l', 'm', 'p', 'pp', 'pvc', 'y', 'cf', 'fg', 'jr', 'lr', 'ma', 'oi',
             'pm', 'ri', 'rm', 'rs', 'sf', 'sm', 'sr', 'ta', 'tc', 'wh', 'cy', 'ap', 'if', 'ih', 'ic', 'tf', 't']

code_dict = {}

for code in code_list:
    code_dict[code] = []
    pattern = re.compile(code + '\d\d\d\d')
    for i in tb_kline_m5.groupby('code'):
        if re.match(pattern, i[0]):
            code_dict[code].append(i[0])

print(code_dict)

for j in tb_kline_m5.groupby('code'):
    if j[0] in code_dict['ag']:
        print(j[1].shape)
        if j[1].shape[0] == 24272:
            sample = j

print(code_dict["ag"])

pure_data = np.array(sample[1]['close'])
data = pure_data[1:] - pure_data[:-1]
# mean_data = data.mean()
# max_data = data.max()
# min_data = data.min()
# data = (data-mean_data)/(max_data-min_data)*10


# --------------------------------------以上为数据提取------------------------------------------#

n_input = 1
n_steps = 50
n_hidden = 12
output_dim = 1

x = tf.placeholder('float', [None, n_steps, n_input])
y = tf.placeholder('float', [None, output_dim])

x1 = tf.unstack(x, n_steps, 1)  # (n_steps,batch_size,n_input)

cell = tf.contrib.rnn.LSTMCell(n_hidden)
tcells = []
for i in range(3):
    tcells.append(tf.contrib.rnn.LSTMCell(n_hidden))
mcell = tf.contrib.rnn.MultiRNNCell(tcells)

outputs, states = tf.contrib.rnn.static_rnn(mcell, x1, dtype=tf.float32)
temp = tf.nn.tanh(outputs)
pred = tf.contrib.layers.fully_connected(outputs[-1], output_dim, activation_fn=None)
##pred = tf.nn.dropout(pred,0.9)
learning_rate = 0.005
training_iters = 3000
batch_size = 500
display_step = 100
loss = tf.reduce_mean(tf.pow(pred - y, 2))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# batch数据准备

length_data = data.shape[0]
X_train = []
Y_train = []

for i in range(round(length_data * 0.8 - n_steps)):
    X_train.append(data[i:i + n_steps])
    Y_train.append(data[i + n_steps])


def batch_data_generate(X_train, Y_train, batch_size):
    index = np.random.choice(len(X_train), size=batch_size, replace=False)
    batch_x = []
    batch_y = []
    for i in index:
        batch_x.append(X_train[i])
        batch_y.append([Y_train[i]])

    return batch_x, batch_y


# 启动session
sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())
step = 1
while step < training_iters:
    batch_x, batch_y = batch_data_generate(X_train, Y_train, batch_size)
    batch_x = np.array(batch_x)
    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
    if step % display_step == 0:
        cost = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
        print('minimatch loss:' + str(cost))
    step += 1

print('finished')

# Test

tag = round(length_data * 0.8)
result = []
X_test = data[(round(length_data * 0.8) - n_steps):round(length_data * 0.8)]
X_test = X_test.reshape(1, n_steps, 1)
feed_dict = {x: X_test}
test_result = data[round(length_data * 0.8) + 1:]
day_count = 0

while tag < length_data - 1:
    feed_dict.update({x: X_test})
    output = sess.run(pred, feed_dict)
    result.append(round(output[0][0]))
    X_test = np.concatenate(((X_test[0][1:]), test_result[day_count].reshape(1, 1)))
    X_test = X_test.reshape(1, n_steps, output_dim)
    tag += 1
    day_count += 1

new_result = result[:]
result = pd.Series(result)
result = -result

## 画图

import matplotlib.pyplot as plt

new_result[0] = new_result[0] + pure_data[round(length_data * 0.8)]
for i in range(1, len(new_result)):
    new_result[i] += new_result[i - 1]

plt.figure()
test_result = pure_data[round(length_data * 0.8) + 1:round(length_data * 0.8) + 1 + len(result)]
x = np.linspace(-1, 1, len(result))
plt.plot(x, test_result, color='red', linestyle='--')
plt.plot(x, new_result)
plt.show()

# 做回测收益曲线


buy_index = result[result > 0].index
sell_index = result[result < 0].index

result = pd.DataFrame(result, columns=['close'])
result.loc[buy_index, 'signal'] = 1
result.loc[sell_index, 'signal'] = 0

result['keep'] = result['signal']
result['keep'] = result['keep'].fillna(method='ffill')
test_result = pd.Series(test_result)
result['benchmark_profit'] = np.log(test_result / test_result.shift(1))
result['benchmark_profit'] = result['benchmark_profit'].fillna(0.0)
result['keep'] = result['keep'].astype('float64')
result['trend_profit'] = result['keep'].values * result['benchmark_profit'].values
result['trend_profit'].plot(figsize=(14, 7))
result[['benchmark_profit', 'trend_profit']].cumsum().plot(grid=True, figsize=(14, 7))
plt.show()
