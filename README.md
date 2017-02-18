# Morphine

**A neural network library built on top of Elixir.**

	neuron = %Morphine.Neuron{inputs: [1, 1], weights: [0.8, 0.2], bias: 0.3}

	Morphine.Neuron.predict(neuron)
	output: 0.55

	smarter = Morphine.Neuron.learn(neuron, 0) # target is zero

	Morphine.Neuron.predict(smarter)
	output: 0.51

	### learn! until it reaches target ###
	genius = Morphine.Neuron.learn!(neuron, 0)

	Morphine.Neuron.predict(genius)
	output: 0

### Author
@leandronsp
