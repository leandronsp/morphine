defmodule Morphine.NeuralNetwork do

  alias Morphine.Layer

  @moduledoc """
  Documentation for Morphine.NeuralNetwork.
  """

  @doc """
  A simple NeuralNetwork.
  """

  def start_link do
    Agent.start_link fn -> %{} end
  end

  def get_layers(network) do
    Agent.get(network, &Map.get(&1, "layers"))
  end

  def put_layer(network, {number_of_neurons, number_of_weights}) do
    layer = build_layer(number_of_neurons, number_of_weights)

    case Agent.get(network, &Map.get(&1, "layers")) do
      nil   -> Agent.update(network, &Map.put(&1, "layers", [layer]))
      found -> Agent.update(network, &Map.put(&1, "layers", found ++ [layer]))
    end
  end

  def setup_layers(network, data) do
    layers = Enum.map(data, fn {number_of_neurons, number_of_weights} ->
      build_layer(number_of_neurons, number_of_weights)
    end)

    Agent.update(network, &Map.put(&1, "layers", layers))
  end

  def update_layers!(network, layers) do
    Agent.update(network, &Map.put(&1, "layers", layers))
  end

  def learn(_, _, _, 0), do: nil

  def learn(network, inputs, outputs, iterations) do
    layers = get_layers(network)

    layer_1 = Enum.at(layers, 0)
    layer_2 = Enum.at(layers, 1)

    {output_layer_1, output_layer_2} = predict(network, inputs)

    sigmoid_derivative = fn array -> Enum.map(array, &sigmoid_derivative(&1)) end
    error_layer_2      = ExAlgebra.Matrix.subtract(outputs, output_layer_2)
    derivative_layer_2 = Enum.map(output_layer_2, &sigmoid_derivative.(&1))
    delta_layer_2      = multiply(error_layer_2, derivative_layer_2)

    error_layer_1      = ExAlgebra.Matrix.multiply(delta_layer_2, ExAlgebra.Matrix.transpose(Layer.to_matrix(layer_2)))
    derivative_layer_1 = Enum.map(output_layer_1, &sigmoid_derivative.(&1))
    delta_layer_1      = multiply(error_layer_1, derivative_layer_1)

    adjustment_layer_1 =
    ExAlgebra.Matrix.transpose(inputs)
    |> ExAlgebra.Matrix.multiply(delta_layer_1)

    adjustment_layer_2 =
    ExAlgebra.Matrix.transpose(output_layer_1)
    |> ExAlgebra.Matrix.multiply(delta_layer_2)

    new_layers = [
      adjust_layer(layer_1, adjustment_layer_1),
      adjust_layer(layer_2, adjustment_layer_2)
    ]

    update_layers!(network, new_layers)
    learn(network, inputs, outputs, iterations - 1)
  end

  def adjust_layer(layer, adjustment) do
    Layer.to_matrix(layer)
    |> ExAlgebra.Matrix.add(adjustment)
    |> Layer.from_matrix
  end

  def predict_result(inputs, layer) do
    activate_func = fn array -> Enum.map(array, &sigmoid(&1)) end
    ExAlgebra.Matrix.multiply(inputs, Layer.to_matrix(layer))
    |> Enum.map(&activate_func.(&1))
  end

  def predict(network, inputs) do
    layers = get_layers(network)
    result = predict_result(inputs, hd(layers))
    predict(network, result, tl(layers), {result})
  end

  def predict(_, _, [], acc), do: acc

  def predict(network, result, inputs, acc) do
    result = predict_result(result, hd(inputs))
    predict(network, result, tl(inputs), Tuple.append(acc, result))
  end

  #### Private functions

  defp build_layer(number_of_neurons, number_of_weights) do
    Layer.build(number_of_neurons, number_of_weights, :random)
  end

  defp sigmoid(calc), do: 1 / (1 + :math.exp(-calc))

  defp sigmoid_derivative(calc), do: calc * (1 - calc)

  defp naive_multiply([], []), do: []

  defp naive_multiply([u_head | u_tail], [v_head | v_tail]) do
    [u_head * v_head | naive_multiply(u_tail, v_tail)]
  end

  defp multiply([], []), do: []

  defp multiply([a_first_row | a_remaining_rows], [b_first_row | b_remaining_rows]) do
    [naive_multiply(a_first_row, b_first_row) | multiply(a_remaining_rows, b_remaining_rows)]
  end

end
