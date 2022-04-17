import numpy as np

w1 = 0.2
w2 = 0.6
bias = 0.2
learning_rate = 0.6
inputs = np.array([[1, 0],
                   [0, 1]])

outputs = np.array([[-1, 1]])


class NET:

    def __init__(self, x1, x2, output):
        self.x1 = x1
        self.x2 = x2
        self.expected_out = output
        self.predicted_out = 0
        self.error = 0

    @staticmethod
    def activation_function(x):
        if x >= 0:
            return 1
        else:
            return -1

    @staticmethod
    def e_error(expected_out, predicted_output):
        return expected_out - predicted_output

    def calculation(self):
        self.net = w1 * self.x1 + w2 * self.x2 + bias
        self.predicted_out = self.activation_function(self.net)

        self.error = self.e_error(self.expected_out, self.predicted_out)

        print("Error E: ", self.error)

        print("Predicted Output: ", self.predicted_out)

        return self.error


def reCalculateWeightsAndBias(self):
    global w1
    global w2
    global bias
    w1 = round((w1 + (learning_rate * self.error * self.x1)), 2)
    print("New w1: ", w1)

    w2 = round((w2 + (learning_rate * self.error * self.x2)), 2)
    print("New w2: ", w2)

    bias = round((bias + (learning_rate * self.error)), 2)
    print("New bias: ", bias)


for i in range(100):
    x11 = inputs[0, 0]
    x12 = inputs[0, 1]
    o1 = outputs[0, 0]
    net_object1 = NET(x11, x12, o1)
    e1 = net_object1.calculation()

    if e1 != 0:
        reCalculateWeightsAndBias(net_object1)

    x21 = inputs[1, 0]
    x22 = inputs[1, 1]
    o2 = outputs[0, 1]
    net_object2 = NET(x21, x22, o2)
    e2 = net_object2.calculation()

    if e2 != 0:
        reCalculateWeightsAndBias(net_object2)

    if e1 == 0 and e2 == 0:
        break

print("w1: ", w1, "w2: ", w2, "bias: ", bias)
