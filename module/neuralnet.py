import numpy as np
import scipy.special

class NeuralNetwork:
    """
    Neural Network
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, *weights):

        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        if weights != ():
            self.weight_ih = weights[0]
            self.weight_ho = weights[1]
        else:
            self.weight_ih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
            self.weight_ho = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))


        self.activation = scipy.special.expit  # sigmoid
        self.inv_activation = scipy.special.logit # logit (inverse of sigmoid)

    def task(self, input_list, sel, target_list=None, learning_rate=0.2):
        """
        train NN or query to predict
        Args:
            input_list (list): set of input values for input layer
            sel : select mode (0 : query, 1 : train)
            target_list (list, optional): set of target values which we finally want

        Returns:
            final_output : result of query
        """
        input_ = np.array(input_list, ndmin=2).T

        hidden_in = np.matmul(self.weight_ih, input_)
        hidden_out = self.activation(hidden_in)

        output_in = np.matmul(self.weight_ho, hidden_out)
        final_output = self.activation(output_in)

        if sel == 0:
            return final_output

        elif sel == 1:

            target_ = np.array(target_list, ndmin=2).T

            output_error = target_ - final_output
            hidden_error = np.matmul(self.weight_ho.T, output_error)

            # Update weights
            self.weight_ho += learning_rate * np.matmul((output_error * final_output * (1.0 - final_output)), hidden_out.T)
            self.weight_ih += learning_rate * np.matmul((hidden_error * hidden_out * (1.0 - hidden_out)), input_.T)

    def backquery(self, label_array):
        """
        query with label instead of image

        Args:
            label_array : array related to a specific label


        Returns:
            bool: The return value. True for success, False otherwise.
        """
        # transpose the targets list to a vertical array
        final_in = np.array(label_array, ndmin=2).T

        # calculate the signal into the final output layer
        final_out = self.inv_activation(final_in)

        # calculate the signal out of the hidden layer
        hidden_in = np.matmul(self.weight_ho.T, final_out)

        hidden_in -= np.min(hidden_in)
        hidden_in /= np.max(hidden_in)
        hidden_in *= 0.98
        hidden_in += 0.01


        hidden_out = self.inv_activation(hidden_in)


        back_output = np.matmul(self.weight_ih.T, hidden_out)

        # scale them back to 0.01 to .99
        back_output -= np.min(back_output)
        back_output /= np.max(back_output)
        back_output *= 0.98
        back_output += 0.01

        return back_output
