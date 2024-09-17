"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use post-hoc RNN to classify original data and synthetic data

Output: discriminative score (np.abs(classification accuracy - 0.5))
"""

# Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from utils import train_test_divide, extract_time, batch_generator
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
df_raw = pd.read_csv("datasets/AirQualityUCI.csv")
border = [0,int(len(df_raw)*0.8),len(df_raw)]
cols_data = df_raw.columns[2:]
df_data = df_raw[cols_data]
data = df_data.values
data_x = data[border[0]:border[1]]
orig_train_data = []
seq_len = 25
length = len(data_x)-seq_len+1
for i in range(length):
    orig_train_data.append(data_x[i:i+seq_len])
synthetic_train_data = np.load("results/aq_mv/train_samples.npy")
scaler = StandardScaler()
scaler.fit(df_data[border[0]:border[1]].values)
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(df_data[border[0]:border[1]].values)

def discriminative_score_metrics (ori_data, generated_data):
  """Use post-hoc RNN to classify original data and synthetic data
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    
  Returns:
    - discriminative_score: np.abs(classification accuracy - 0.5)
  """
  # Initialization on the Graph
  tf.reset_default_graph()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape    
    
  # Set maximum sequence length and each sequence length
  ori_time, ori_max_seq_len = extract_time(ori_data)
  generated_time, generated_max_seq_len = extract_time(ori_data)
  max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
     
  ## Builde a post-hoc RNN discriminator network 
  # Network parameters
  hidden_dim = int(dim/2)
  iterations = 2000
  batch_size = 128
    
  # Input place holders
  # Feature
  X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
  X_hat = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x_hat")
    
  T = tf.placeholder(tf.int32, [None], name = "myinput_t")
  T_hat = tf.placeholder(tf.int32, [None], name = "myinput_t_hat")
    
  # discriminator function
  def discriminator (x, t):
    """Simple discriminator function.
    
    Args:
      - x: time-series data
      - t: time information
      
    Returns:
      - y_hat_logit: logits of the discriminator output
      - y_hat: discriminator output
      - d_vars: discriminator variables
    """
    with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE) as vs:
      d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')
      d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, x, dtype=tf.float32, sequence_length = t)
      y_hat_logit = tf.contrib.layers.fully_connected(d_last_states, 1, activation_fn=None) 
      y_hat = tf.nn.sigmoid(y_hat_logit)
      d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
    
    return y_hat_logit, y_hat, d_vars
    
  y_logit_real, y_pred_real, d_vars = discriminator(X, T)
  y_logit_fake, y_pred_fake, _ = discriminator(X_hat, T_hat)
        
  # Loss for the discriminator
  d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_real, 
                                                                       labels = tf.ones_like(y_logit_real)))
  d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_fake, 
                                                                       labels = tf.zeros_like(y_logit_fake)))
  d_loss = d_loss_real + d_loss_fake
    
  # optimizer
  d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list = d_vars)
        
  ## Train the discriminator   
  # Start session and initialize
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
    
  # Train/test division for both original and generated data
  train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
  train_test_divide(ori_data, generated_data, ori_time, generated_time)
    
  # Training step
  for itt in range(iterations):
          
    # Batch setting
    X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
    X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
          
    # Train discriminator
    _, step_d_loss = sess.run([d_solver, d_loss], 
                              feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb})            
  
  ## Test the performance on the testing set    
  y_pred_real_curr, y_pred_fake_curr = sess.run([y_pred_real, y_pred_fake], 
                                                feed_dict={X: test_x, T: test_t, X_hat: test_x_hat, T_hat: test_t_hat})
    
  y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis = 0))
  y_label_final = np.concatenate((np.ones([len(y_pred_real_curr),]), np.zeros([len(y_pred_fake_curr),])), axis = 0)
    
  # Compute the accuracy
  acc = accuracy_score(y_label_final, (y_pred_final>0.5))
  discriminative_score = np.abs(0.5-acc)
  print('Accuracy: ' + str(acc), 'Discriminative Score: ' + str(discriminative_score))
  data_x = data[border[1]:border[2]]
  orig_test_data = []
  seq_len = 25
  length = len(data_x)-seq_len+1
  for i in range(length):
      orig_test_data.append(data_x[i:i+seq_len])
  synthetic_test_data = np.load("results/aq_mv/test_samples.npy")
  normalized_orig_data = []
  for d in orig_test_data:
      d = minmax_scaler.transform(d)
      normalized_orig_data.append(d)
  normalized_synthetic_data = []
  for d in synthetic_test_data:
      d = minmax_scaler.transform(scaler.inverse_transform(d))
      normalized_synthetic_data.append(d)
  test_x = normalized_orig_data
  test_x_hat = normalized_synthetic_data
  test_t = extract_time(test_x)[0]
  test_t_hat = extract_time(test_x_hat)[0]
  ## Test the performance on the testing set    
  y_pred_real_curr, y_pred_fake_curr = sess.run([y_pred_real, y_pred_fake], 
                                                feed_dict={X: test_x, T: test_t, X_hat: test_x_hat, T_hat: test_t_hat})
    
  y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis = 0))
  y_label_final = np.concatenate((np.ones([len(y_pred_real_curr),]), np.zeros([len(y_pred_fake_curr),])), axis = 0)
    
  # Compute the accuracy
  acc = accuracy_score(y_label_final, (y_pred_final>0.5))
  discriminative_score = np.abs(0.5-acc)
  print('Accuracy: ' + str(acc), 'Discriminative Score: ' + str(discriminative_score))
  return discriminative_score  

if __name__ == '__main__':
  normalized_orig_data = []
  for d in orig_train_data:
      d = minmax_scaler.transform(d)
      normalized_orig_data.append(d)
  normalized_synthetic_data = []
  for d in synthetic_train_data:
      d = minmax_scaler.transform(scaler.inverse_transform(d))
      normalized_synthetic_data.append(d)
  for i in range(5):
    discriminative_score_metrics(normalized_orig_data,normalized_synthetic_data)