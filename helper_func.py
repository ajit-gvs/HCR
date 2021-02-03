def create_placeholders(input_H, input_W, no_channels, no_classes):

    X = tf.placeholder(tf.float32, shape=(None, input_H, input_W, no_channels), name="X")
    Y = tf.placeholder(tf.float32, shape=(None, no_classes), name="Y")
    
    return X, Y

def initialize_parameters():
    
    W1 = tf.get_variable("W1", [3, 3, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [2, 2, 32, 32], initializer = tf.contrib.layers.xavier_initializer())

    parameters = {"W1": W1,
                  "W2": W2}
   
    return parameters

def buildCNN(input):
  
  kernel1 = tf.Variable(tf.random.truncated_normal([5, 5, 1, 32,stddev=0.1))
  #Convolution Layer1
  C1 = tf.nn.conv2d(input,kernel, strides = [1,1,1,1], padding = 'SAME')
  #Activation Layer1
  A1 = tf.nn.relu(C1)
  #Batch Normalization Layer1
  BN1 = tf.compat.v1.layers.batch_normalization(A1, training=self.is_train)
  #Max Pooling Layer1
  P1 = tf.nn.max_pool(BN1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')


  kernel2 = tf.Variable(tf.random.truncated_normal(5, 5, 32, 64,stddev=0.1))
  #Convolution Layer2
  C2 = tf.nn.conv2d(P1,kernel2, strides = [1,1,1,1], padding = 'SAME')
  #Activation Layer2
  A2 = tf.nn.relu(C2)
  #Batch Normalization Layer2
  BN2 = tf.compat.v1.layers.batch_normalization(A2, training=self.is_train)
  #Max Pooling Layer2
  P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')


  kernel3 = tf.Variable(tf.random.truncated_normal(3, 3, 64, 128,stddev=0.1))
  #Convolution Layer3
  C3 = tf.nn.conv2d(P2,kernel3, strides = [1,1,1,1], padding = 'SAME')
  #Activation Layer3
  A3 = tf.nn.relu(C3)
  #Batch Normalization Layer2
  BN3 = tf.compat.v1.layers.batch_normalization(A3, training=self.is_train)
  #Max Pooling Layer2
  P3 = tf.nn.max_pool(A3, ksize = [1,1,2,1], strides = [1,1,2,1], padding = 'VALID')


  # FLATTEN
  F = tf.contrib.layers.flatten(P3)
  Dense1 = tf.contrib.layers.fully_connected(P2, CLASSES, activation_fn=None)
  Activation1=tf.nn.relu(Dense1)
  Batch_norm1 = tf.compat.v1.layers.batch_normalization(Activation1, training=self.is_train)


  output = tf.contrib.layers.fully_connected(Batch_norm1, CLASSES, activation_fn=None)
  return output
  

  

def compute_cost(output, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = Y))
    
    return cost

def random_batches(X, Y, batch_size = 64):
    # number of training examples
    no_samples = X.shape[0]                  
    batches = []
    
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(no_samples))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Partition (shuffled_X, shuffled_Y) and remove the end case.
    num_complete_batches = math.floor(no_samples/batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for i in range(0, num_complete_batches):
        batch_X = shuffled_X[i * batch_size : i * batch_size + batch_size,:,:,:]
        batch_Y = shuffled_Y[i * batch_size : i * batch_size + batch_size,:]
        batch = (batch_X, batch_Y)
        batches.append(batch)
    
    # Handling the end case if last mini-batch < batch_size
    if no_samples % batch_size != 0:
        batch_X = shuffled_X[num_complete_batches * batch_size : no_samples,:,:,:]
        batch_Y = shuffled_Y[num_complete_batches * batch_size : no_samples,:]
        batch = (batch_X, batch_Y)
        batches.append(batch)
    
    return mini_batches
  

  

  

      

   