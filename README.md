# Morphine

**A neural network library built on top of Elixir.**

Inspired by [Mind](https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/) and this [awesome blog post](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1#.voyy4g51x)

### Api examples

	alias Morphine.NeuralNetwork, as: Network

	### XOR gate
	inputs  = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
	outputs = Morphing.Calc.Matrix.transpose([[0, 1, 1, 1, 1, 0, 0]])

	{:ok, network} = Network.start_link
	Network.setup_layers(network, [{4, 3}, {1, 4} # {number_of_neurons, number_of_weights}])

	Network.learn(network, inputs, outputs, 100000)

	output = Network.predict(network, [[1, 1, 0]])
	# output ~ [[0.007]]

	output = Network.predict(network, [[1, 0, 2]])
	# output ~ [[0.994]]

### Development (using Docker)
```
make mix.deps.get
make mix.test
```

### Author
@leandronsp
