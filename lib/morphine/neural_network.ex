defmodule Morphine.NeuralNetwork do
  @moduledoc """
  A Network is the core of Morphine. It sets up layers and persist them by using
  Agents.

  Layers can be updated anytime.
  """

  alias Morphine.Layer
  alias Morphine.BackPropagation, as: BackPropagation
  alias Morphine.ForwardPropagation, as: ForwardPropagation

  @type neuron  :: Morphine.Neuron
  @type layer   :: Morphine.Layer
  @type network :: Morphine.NeuralNetwork

  def start_link do
    Agent.start_link fn -> %{} end
  end

  @doc """
  ## Examples

      iex> {:ok, network} = Morphine.NeuralNetwork.start_link
      iex> Morphine.NeuralNetwork.get_layers(network)
      nil
  """

  @spec get_layers(network) :: list
  def get_layers(network) do
    Agent.get(network, &Map.get(&1, "layers"))
  end

  @doc """
  Given a number of neurons and the number of weights for each neuron, creates a
  persistent layer and appends to the list of layers.

  ## Examples

      iex> {:ok, network} = Morphine.NeuralNetwork.start_link
      iex> Morphine.NeuralNetwork.put_layer(network, {1, 4})
      iex> Morphine.NeuralNetwork.get_layers(network) |> length
      1
  """

  @spec put_layer(network, tuple) :: any
  def put_layer(network, {number_of_neurons, number_of_weights}) do
    layer = build_layer(number_of_neurons, number_of_weights)

    case Agent.get(network, &Map.get(&1, "layers")) do
      nil   -> Agent.update(network, &Map.put(&1, "layers", [layer]))
      found -> Agent.update(network, &Map.put(&1, "layers", found ++ [layer]))
    end
  end

  @doc """
  Creates persistent layers given an array of tuples containing:
    - number of neurons
    - number of weights for each neuron

  Please keep in mind that this operation will OVERRIDE the current persistent layers.

  ## Examples

      iex> {:ok, network} = Morphine.NeuralNetwork.start_link
      iex> Morphine.NeuralNetwork.setup_layers(network, [{1, 4}, {4, 3}])
      iex> Morphine.NeuralNetwork.get_layers(network) |> length
      2
  """

  @spec setup_layers(network, list) :: any
  def setup_layers(network, data) do
    layers = Enum.map(data, fn {number_of_neurons, number_of_weights} ->
      build_layer(number_of_neurons, number_of_weights)
    end)

    Agent.update(network, &Map.put(&1, "layers", layers))
  end

  @doc """
  Same as `setup_layers`, but this operation takes a list of layers instead.
  Keep in mind that this operation will OVERRIDE the current persistent layers.

  ## Examples

      iex> {:ok, network} = Morphine.NeuralNetwork.start_link
      iex> Morphine.NeuralNetwork.update_layers!(network, [%Morphine.Layer{neurons: []}])
      iex> Morphine.NeuralNetwork.get_layers(network) |> length
      1
  """

  @spec update_layers!(network, list) :: any
  def update_layers!(network, layers) do
    Agent.update(network, &Map.put(&1, "layers", layers))
  end

  @doc """
  Network learning involves forward propagation, back propagation
  with adjustments, then updating persistent layers.

  This process is repeated N times accordingly number of iterations.

  ## Examples

      iex> {:ok, network} = Morphine.NeuralNetwork.start_link
      iex> Morphine.NeuralNetwork.setup_layers(network, [{4, 3}, {4, 4}, {1, 4}])
      iex> inputs  = [[0, 0, 1], [0, 1, 1], [1, 0, 1]]
      iex> outputs = Morphine.Calc.Matrix.transpose([[0, 1, 1]])
      iex> Morphine.NeuralNetwork.learn(network, inputs, outputs, 10000)
      iex> result = Morphine.NeuralNetwork.predict!(network, [[1, 1, 0]])
      iex> result > 0.99
      true
  """

  @spec learn(network, list, list, number) :: any
  def learn(_, _, _, 0), do: nil
  def learn(network, inputs, targets, iterations) do
    layers             = get_layers(network)
    outputs            = ForwardPropagation.forward(layers, inputs)
    layers_and_outputs = Enum.zip(layers, outputs)
    new_layers         = BackPropagation.adjust(inputs, targets, layers_and_outputs)

    update_layers!(network, new_layers)
    learn(network, inputs, targets, iterations - 1)
  end

  @doc """
  Calculates the output.

  ## Examples

      iex> {:ok, network} = Morphine.NeuralNetwork.start_link
      iex> Morphine.NeuralNetwork.setup_layers(network, [{4, 3}, {4, 4}, {1, 4}])
      iex> result = Morphine.NeuralNetwork.predict!(network, [[1, 1, 0]])
      iex> result < 1.0
      true
  """

  @spec predict!(network, list) :: number
  def predict!(network, inputs) do
    get_layers(network) |> ForwardPropagation.predict!(inputs)
  end

  @spec build_layer(integer, integer) :: layer
  defp build_layer(number_of_neurons, number_of_weights) do
    Layer.build(number_of_neurons, number_of_weights, :random)
  end
end
