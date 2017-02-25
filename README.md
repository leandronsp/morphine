# Morphine

**A neural network library built on top of Elixir.**

### Api examples

	### XOR gate
	inputs = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
	outputs = ExAlgebra.Matrix.transpose([[0, 1, 1, 1, 1, 0, 0]]) 

	alias Morphine.NeuralNetwork, as: Network
	
	{:ok, network} = Network.start_link		
	Network.setup_layers(network, [{4, 3}, {1, 4} # {number_of_neurons, number_of_weights}])
	
	Network.learn(network, inputs, outputs, 100000)
	
	{_, output} = Network.predict(network, [[1, 1, 0]]) 
	# output ~ [[0.007]]
	
	{_, output} = Network.predict(network, [[1, 0, 2]]) 
	# output ~ [[0.994]]

### Author
@leandronsp
