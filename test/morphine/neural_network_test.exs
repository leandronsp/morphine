require IEx

defmodule Morphine.NeuralNetworkTest do
  use ExUnit.Case, async: true
  doctest Morphine.NeuralNetwork

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
    def train(_, weights, _, 0), do: weights

    def train(inputs, weights, outputs, iterations) do
      output = predict(weights, inputs)
      IO.puts "output"
      IO.inspect output
      error  = ExAlgebra.Matrix.subtract(outputs, output)
      IO.puts "error"
      IO.inspect error

      func = fn array -> Enum.map(array, &ActivateFunction.sigmoid_derivative(&1)) end
      derivative = Enum.map(output, &func.(&1))
      IO.puts "derivative"
      IO.inspect derivative

      multi = Helper.multiply(error, derivative)

      IO.puts "multi"
      IO.inspect multi

      tinputs = ExAlgebra.Matrix.transpose(inputs)
      IO.puts "tinputs"
      IO.inspect tinputs
      adjustment = ExAlgebra.Matrix.multiply(tinputs, multi)

      IO.puts "adjustment"
      IO.inspect adjustment
      train(inputs, ExAlgebra.Matrix.add(weights, adjustment), outputs, iterations - 1)
      #build_neuron(ExAlgebra.Matrix.add(neuron.weights, adjustment))
    end

    def predict(weights, inputs) do
      func = fn array -> Enum.map(array, &ActivateFunction.sigmoid(&1)) end
      ExAlgebra.Matrix.multiply(inputs, weights)
      |> Enum.map(&func.(&1))
    end

    def build_neuron(weights, bias \\ 1) do
      %Neuron{weights: weights, bias: bias}
    end
  end

  test "start" do
    inputs  = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
    weights = [[-0.16595599], [0.44064899], [-0.99977125]]
    outputs = ExAlgebra.Matrix.transpose([[0, 1, 1, 0]])

    new_weights = Network.train(inputs, weights, outputs, 10000)
    assert new_weights == [[9.672993027737387], [-0.20784349926468884], [-4.629636687732811]]

    assert Network.predict(new_weights, [[1, 0, 0]]) == [[0.9999370428352157]]
  end
end
