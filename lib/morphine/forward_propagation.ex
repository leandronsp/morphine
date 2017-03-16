defmodule Morphine.ForwardPropagation do
  @moduledoc """
  Calculates the output across layers.
  """

  alias Morphine.Layer
  alias Morphine.Calc.Activation, as: Activation
  alias Morphine.Calc.Matrix, as: Matrix

  @doc """
  Forward and predict returning a single output.

  ## Examples

    iex> {:ok, network} = Morphine.NeuralNetwork.start_link
    iex> Morphine.NeuralNetwork.setup_layers(network, [{4, 3}, {4, 4}, {1, 4}])
    iex> layers = Morphine.NeuralNetwork.get_layers(network)
    iex> result = Morphine.ForwardPropagation.predict!(layers, [[1, 1, 0]])
    iex> result < 1.0
    true
  """

  @spec predict!(list, list) :: number
  def predict!(layers, inputs) do
    forward(layers, inputs) |> List.flatten |> List.last
  end

  @doc """
  Forward propagation thru layers and return their respective outputs.

  ## Examples

    iex> {:ok, network} = Morphine.NeuralNetwork.start_link
    iex> Morphine.NeuralNetwork.setup_layers(network, [{4, 3}, {4, 4}, {1, 4}])
    iex> layers = Morphine.NeuralNetwork.get_layers(network)
    iex> result = Morphine.ForwardPropagation.forward(layers, [[1, 1, 0]])
    iex> result |> List.flatten |> length
    9
  """

  @spec forward(list, list) :: list
  def forward([], inputs), do: inputs
  def forward(layers, inputs) do
    result = predict_result(inputs, hd(layers))
    predict(result, tl(layers), {result})
    |> Tuple.to_list
  end

  defp predict(_, [], acc), do: acc
  defp predict(result, inputs, acc) do
    result = predict_result(result, hd(inputs))
    predict(result, tl(inputs), Tuple.append(acc, result))
  end

  defp predict_result(inputs, layer) do
    activate_func = fn array -> Enum.map(array, &Activation.sigmoid(&1)) end
    Matrix.multiply(inputs, Layer.to_matrix(layer))
    |> Enum.map(&activate_func.(&1))
  end

end
