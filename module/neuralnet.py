import numpy as np
import scipy.special

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, *weights):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        self.l_rate = learning_rate
        self.weight_ih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.weight_ho = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        self.activation = scipy.special.expit  # sigmoid

    def task(self, input_list, sel, target_list=None):
        """
        train NN or query to predict
        Args:
            input_list (list): set of input values for input layer
            sel : select mode (0 : query, 1 : train)
            target_list (list, optional): set of target values which we finally want

        Returns:
            bool: The return value. True for success, False otherwise.
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
            self.weight_ho += self.l_rate * np.matmul((output_error * final_output * (1.0 - final_output)), hidden_out.T)
            self.weight_ih += self.l_rate * np.matmul((hidden_error * hidden_out * (1.0 - hidden_out)), input_.T)


    # def query(self, input_list):
    #     """
    #     calculate Neural Network's output through weights
    #     Args:
    #         input_list (list) : input values of Neural Network
    #     Returns:
    #         final_output (list) : Output values of Neural Network
    #     """
    #     input_ = np.array(input_list, ndmin=2).T

    #     hidden_in = np.matmul(self.weight_ih, input_)
    #     hidden_out = self.activation(hidden_in)

    #     output_in = np.matmul(self.weight_ho, hidden_out)
    #     final_output = self.activation(output_in)

    #     return final_output
