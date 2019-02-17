import numpy as np



class raw_rnn:

    def __init__(self,sequence_vector,state_vector):

        #[sequence_len,var_len]
        self.sequence_vector=sequence_vector
        self.state_vector=state_vector
        self.state_len=len(state_vector)
        self.sequence_len=sequence_vector.shape[0]
        self.var_len=sequence_vector.shape[1]
        self.weight_cell=np.ndarray(shape=(self.sequence_len,self.var_len+self.state_len,self.state_len))
        self.bias_cell=np.ndarray(shape=(self.sequence_len,self.state_len))
        self.weight_ouput=np.ndarray(shape= (self.sequence_len,self.state_len))
        self.bias_ouput=np.ndarray(shape=(self.sequence_len,1))
        self.o=np.ndarray(shape=(self.sequence_len,))

    def f_brocast(self):
        for i in range(self.sequence_len):
            self.weight_cell[i] = np.random.rand(self.var_len + self.state_len, self.state_len)
            self.bias_cell[i] = np.random.rand(self.state_len)
            self.weight_ouput[i] = np.random.rand(self.state_len)
            self.bias_ouput[i] = np.random.rand(1)
            #[1,var_len] [1,state_len]
            input_vector=np.r_[self.sequence_vector[i],self.state_vector]
            self.state_vector=self.tanh(np.dot(input_vector,self.weight_cell[i])+self.bias_cell[i])
            self.o[i]=np.dot(self.state_vector,self.weight_ouput[i])+self.bias_ouput[i]
        return  self.o

    def sigmoid(self,vector):
        return vector

    def tanh(self,vector):
        return np.tanh(vector)


if __name__=='__main__':


    sequence_vector=np.array([[1,2,3],[4,5,6],[6,7,8],[9,10,11]])

    state_vector=np.array([0,0])

    rnn=raw_rnn(sequence_vector,state_vector)

    o=rnn.f_brocast()

    print(o)

