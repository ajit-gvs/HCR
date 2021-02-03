def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001,
          num_epochs = 100, minibatch_size = 64, print_cost = True):

    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1] 
    
    
    costs = []                                        
    
    # Createing Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

     # Forward propagation
    output = buildCNN(X)
    
    # Cost function
    cost = compute_cost(output, Y)
    
    # Backpropagation
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initializing all the variables globally
    init = tf.global_variables_initializer()
     
    # Starting the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Doing the training loop
        for epoch in range(num_epochs):

            batch_cost = 0.
            # number of batches of size batch_size in the train set
            num_batches = int(no_samples / batch_size)
            batches = random_batches(X_train, Y_train,batch_size)

            for batch in batches:

                # Select a minibatch
                (batch_X, batch_Y) = batch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: batch_X, Y: batch_Y})
                
                batch_cost += temp_cost / num_batches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")     
        
        # Calculate the correct predictions
        predict_op = tf.argmax(output, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        
        # Saving our trained model
        saver = tf.train.Saver()
        tf.add_to_collection('predict_op', predict_op)
        saver.save(sess, './my-CNN-model')        
                
        return train_accuracy, test_accuracy, parameters