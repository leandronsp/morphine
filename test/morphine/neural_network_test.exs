require IEx

defmodule Morphine.NeuralNetworkTest do
  use ExUnit.Case, async: true
  doctest Morphine.NeuralNetwork

  defmodule Layer do
    defstruct weights: []
  end

  defmodule Neuron do
    defstruct weights: [], bias: 1
  end

  defmodule Helper do
    def naive_multiply([], []), do: []

    def naive_multiply([u_head | u_tail], [v_head | v_tail]) do
      [u_head * v_head | naive_multiply(u_tail, v_tail)]
    end

    def multiply([], []), do: []

    def multiply([a_first_row | a_remaining_rows], [b_first_row | b_remaining_rows]) do
      [naive_multiply(a_first_row, b_first_row) | multiply(a_remaining_rows, b_remaining_rows)]
    end
  end

  defmodule ActivateFunction do
    def sigmoid(calc), do: 1 / (1 + :math.exp(-calc))
    def sigmoid_derivative(calc), do: calc * (1 - calc)
  end

  defmodule Network do
    def train(_, layers, _, 0), do: layers

    def train(inputs, layers, outputs, iterations) do
      layer_1 = Enum.at(layers, 0)
      layer_2 = Enum.at(layers, 1)
      {output_layer_1, output_layer_2} = Network.predict(inputs, layers, {})

      sigmoid_derivative = fn array -> Enum.map(array, &ActivateFunction.sigmoid_derivative(&1)) end
      error_layer_2      = ExAlgebra.Matrix.subtract(outputs, output_layer_2)
      derivative_layer_2 = Enum.map(output_layer_2, &sigmoid_derivative.(&1))
      delta_layer_2      = Helper.multiply(error_layer_2, derivative_layer_2)

      error_layer_1      = ExAlgebra.Matrix.multiply(delta_layer_2, ExAlgebra.Matrix.transpose(layer_2.weights))
      derivative_layer_1 = Enum.map(output_layer_1, &sigmoid_derivative.(&1))
      delta_layer_1      = Helper.multiply(error_layer_1, derivative_layer_1)

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

      train(inputs, new_layers, outputs, iterations - 1)
    end

    def adjust_layer(layer, adjustment) do
      %Layer{weights: ExAlgebra.Matrix.add(layer.weights, adjustment)}
    end

    def predict(inputs, [hlayer|tlayers], acc) do
      activate_func = fn array -> Enum.map(array, &ActivateFunction.sigmoid(&1)) end

      result = ExAlgebra.Matrix.multiply(inputs, hlayer.weights)
      |> Enum.map(&activate_func.(&1))

      predict(result, tlayers, Tuple.append(acc, result))
    end

    def predict(_, [], acc), do: acc
  end

  test "start" do
    layer_1 = %Layer{weights: [[-0.16595599, 0.44064899, -0.99977125, -0.39533485],
       [-0.70648822, -0.81532281, -0.62747958, -0.30887855],
       [-0.20646505, 0.07763347, -0.16161097, 0.370439]]}

    layer_2 = %Layer{weights: [[-0.5910955],
       [0.75623487],
       [-0.94522481],
       [0.34093502]]}

    inputs = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
    outputs = ExAlgebra.Matrix.transpose([[0, 1, 1, 1, 1, 0, 0]])

    layers = [layer_1, layer_2]
    smarter_layers = Network.train(inputs, layers, outputs, 60000)
    {_, output} = Network.predict([[1, 1, 0]], smarter_layers, {})
    assert output == [[0.007887604373626915]]
  end
end
