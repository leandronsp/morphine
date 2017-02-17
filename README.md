# Morphine

**A neural network library built on top of Elixir.**

	use Morphine.Neuron
	
	%Neuron{inputs: [1, 0], weights: [0.5, 0.8], bias: 0.2}
	|> Neuron.calculate_output
	|> Neuron.calculate_error_margin(1)

	output: 0.223427890212	

### Author
@leandronsp