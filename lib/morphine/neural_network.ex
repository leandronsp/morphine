require IEx

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

  def delta(target, output) do
    func  = fn array -> Enum.map(array, &sigmoid_derivative(&1)) end
    error = ExAlgebra.Matrix.subtract(target, output)
    Enum.map(output, &func.(&1)) |> multiply(error)
  end

  def delta(output, next_layer, delta_next_layer) do
    func  = fn array -> Enum.map(array, &sigmoid_derivative(&1)) end
    factor = ExAlgebra.Matrix.transpose(Layer.to_matrix(next_layer))
    error  = ExAlgebra.Matrix.multiply(delta_next_layer, factor)

    Enum.map(output, &func.(&1)) |> multiply(error)
  end

  def adjust(inputs, target, [{layer, output}]) do
    delta      = delta(target, output)
    adjustment = ExAlgebra.Matrix.transpose(inputs) |> ExAlgebra.Matrix.multiply(delta)

    adjusted =
    Layer.to_matrix(layer)
    |> ExAlgebra.Matrix.add(adjustment)
    |> Layer.from_matrix

    [adjusted]
  end

  def adjust(inputs, target, [{layer, output}|remaining]) do
    {{next_layer, output_next_layer}, _} = List.pop_at(remaining, 0)

    deltanl    = delta(target, output_next_layer)
    delta      = delta(output, next_layer, deltanl)
    adjustment = ExAlgebra.Matrix.transpose(inputs) |> ExAlgebra.Matrix.multiply(delta)

    adjusted =
    Layer.to_matrix(layer)
    |> ExAlgebra.Matrix.add(adjustment)
    |> Layer.from_matrix

    adjust(inputs, target, remaining, output, [adjusted])
  end

  def adjust(_, target, [{layer, output}|[]], output_previous_layer, acc) do
    delta      = delta(target, output)
    adjustment = ExAlgebra.Matrix.transpose(output_previous_layer) |> ExAlgebra.Matrix.multiply(delta)

    adjusted =
    Layer.to_matrix(layer)
    |> ExAlgebra.Matrix.add(adjustment)
    |> Layer.from_matrix

    acc ++ [adjusted]
  end

  def adjust(inputs, target, [{layer, output}|remaining], output_previous_layer, acc) do
    {{next_layer, output_next_layer}, _} = List.pop_at(remaining, 0)

    deltanl    = delta(target, output_next_layer)
    delta      = delta(output, next_layer, deltanl)
    adjustment = ExAlgebra.Matrix.transpose(output_previous_layer) |> ExAlgebra.Matrix.multiply(delta)

    adjusted =
    Layer.to_matrix(layer)
    |> ExAlgebra.Matrix.add(adjustment)
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

  def predict(network, inputs) do
    layers = get_layers(network)
    result = predict_result(inputs, hd(layers))
    predict(result, tl(layers), {result})
  end

  #### Private functions

  defp predict( _, [], acc), do: acc

  defp predict(result, inputs, acc) do
    result = predict_result(result, hd(inputs))
    predict(result, tl(inputs), Tuple.append(acc, result))
  end

  defp predict_result(inputs, layer) do
    activate_func = fn array -> Enum.map(array, &sigmoid(&1)) end
    ExAlgebra.Matrix.multiply(inputs, Layer.to_matrix(layer))
    |> Enum.map(&activate_func.(&1))
  end

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
