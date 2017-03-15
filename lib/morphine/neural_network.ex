defmodule Morphine.NeuralNetwork do
  @moduledoc """
  A Network is the core of Morphine. It sets up layers and persist them by using
  Agents.

  Layers can be updated anytime.
  """

  alias Morphine.Layer
  alias Morphine.Calc.Matrix, as: Matrix

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

  def delta(target, output) do
    func  = fn array -> Enum.map(array, &sigmoid_derivative(&1)) end
    error = Matrix.subtract(target, output)
    Enum.map(output, &func.(&1)) |> Matrix.naive_multiply(error)
  end

  def delta(output, next_layer, delta_next_layer) do
    func  = fn array -> Enum.map(array, &sigmoid_derivative(&1)) end
    factor = Matrix.transpose(Layer.to_matrix(next_layer))
    error  = Matrix.multiply(delta_next_layer, factor)

    Enum.map(output, &func.(&1)) |> Matrix.naive_multiply(error)
  end

  def adjust(inputs, target, [{layer, output}]) do
    delta      = delta(target, output)
    adjustment = Matrix.transpose(inputs) |> Matrix.multiply(delta)

    adjusted =
    Layer.to_matrix(layer)
    |> Matrix.add(adjustment)
    |> Layer.from_matrix

    [adjusted]
  end

  def adjust(inputs, target, [{layer, output}|remaining]) do
    {{last_layer, output_last_layer}, _} = List.pop_at(remaining, length(remaining) - 1)

    deltanl    = delta(target, output_last_layer)
    delta      = delta(output, last_layer, deltanl)
    adjustment = Matrix.transpose(inputs) |> Matrix.multiply(delta)

    adjusted =
    Layer.to_matrix(layer)
    |> Matrix.add(adjustment)
    |> Layer.from_matrix

    adjust(inputs, target, remaining, output, [adjusted])
  end

  def adjust(_, target, [{layer, output}|[]], output_previous_layer, acc) do
    delta      = delta(target, output)
    adjustment = Matrix.transpose(output_previous_layer) |> Matrix.multiply(delta)

    adjusted =
    Layer.to_matrix(layer)
    |> Matrix.add(adjustment)
    |> Layer.from_matrix

    acc ++ [adjusted]
  end

  def adjust(inputs, target, [{layer, output}|remaining], output_previous_layer, acc) do
    {{last_layer, output_last_layer}, _} = List.pop_at(remaining, length(remaining) - 1)

    deltanl    = delta(target, output_last_layer)
    delta      = delta(output, last_layer, deltanl)
    adjustment = Matrix.transpose(output_previous_layer) |> Matrix.multiply(delta)

    adjusted =
    Layer.to_matrix(layer)
    |> Matrix.add(adjustment)
    |> Layer.from_matrix

    adjust(inputs, target, remaining, output, acc ++ [adjusted])
  end

  def learn(_, _, _, 0), do: nil

  def learn(network, inputs, targets, iterations) do
    layers             = get_layers(network)
    outputs            = predict(network, inputs) |> Tuple.to_list
    layers_and_outputs = Enum.zip(layers, outputs)
    new_layers         = adjust(inputs, targets, layers_and_outputs)

    update_layers!(network, new_layers)
    learn(network, inputs, targets, iterations - 1)
  end

  def predict!(_, result, []) do
    Enum.reduce(result, 1, fn arr, acc ->
      acc * Enum.reduce(arr, 1, &*/2)
    end)
  end

  def predict!(network, result, inputs) do
    result = predict_result(result, hd(inputs))
    predict!(network, result, tl(inputs))
  end

  def predict!(network, inputs) do
    layers = get_layers(network)
    result = predict_result(inputs, hd(layers))
    predict!(network, result, tl(layers))
  end

  #### Private functions

  defp predict(network, inputs) do
    layers = get_layers(network)
    result = predict_result(inputs, hd(layers))
    predict(result, tl(layers), {result})
  end

  defp predict( _, [], acc), do: acc

  defp predict(result, inputs, acc) do
    result = predict_result(result, hd(inputs))
    predict(result, tl(inputs), Tuple.append(acc, result))
  end

  defp predict_result(inputs, layer) do
    activate_func = fn array -> Enum.map(array, &sigmoid(&1)) end
    Matrix.multiply(inputs, Layer.to_matrix(layer))
    |> Enum.map(&activate_func.(&1))
  end

  defp build_layer(number_of_neurons, number_of_weights) do
    Layer.build(number_of_neurons, number_of_weights, :random)
  end

  defp sigmoid(calc), do: 1 / (1 + :math.exp(-calc))

  defp sigmoid_derivative(calc), do: calc * (1 - calc)
end
