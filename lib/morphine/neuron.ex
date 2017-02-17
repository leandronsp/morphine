defmodule Morphine.Neuron do
  defstruct inputs: [], weights: [], bias: 0

  @moduledoc """
  Documentation for Morphine.Neuron.
  """

  @doc """
  A simple Neuron.

  ## Examples

      iex> Morphine.Neuron.summation(%{inputs: [1, 0.4, 0.2], weights: [0.2, 0.8, 0.3]})
      0.5800000000000001

      iex> Morphine.Neuron.sigmoid(0.5800000000000001)
      0.6410674063348171

      iex> Morphine.Neuron.add_bias(10, %Morphine.Neuron{bias: 42})
      52

      iex> neuron = %Morphine.Neuron{inputs: [1, 0.4, 0.2], weights: [0.2, 0.8, 0.3]}
      iex> Morphine.Neuron.calculate_output(neuron)
      0.6410674063348171

      iex> neuron = %Morphine.Neuron{inputs: [1, 0.4, 0.2], weights: [0.2, 0.8, 0.3], bias: 0.45}
      iex> Morphine.Neuron.calculate_output(neuron)
      0.7369158958334202

      iex> Morphine.Neuron.calculate_error_margin(0.8, 1)
      0.19999999999999996

      iex> Morphine.Neuron.sigmoid_prime(1.235)
      0.17454403433618204

      iex> Morphine.Neuron.calculate_delta_output(1.235, -0.77)
      -0.13439890643886018

  """

  def summation(%{inputs: [], weights: []}), do: 0

  def summation(%{inputs: [h1|t1], weights: [h2|t2]}) do
    h1 * h2 + summation(%{inputs: t1, weights: t2})
  end

  def add_bias(sum, neuron), do: sum + neuron.bias

  def sigmoid(sum), do: 1 / (1 + :math.exp(-sum))

  def sigmoid_prime(sum) do
    :math.exp(-sum) / :math.pow(1 + :math.exp(-sum), 2)
  end

  def calculate_output(neuron) do
    summation(neuron)
    |> add_bias(neuron)
    |> sigmoid
  end

  def calculate_error_margin(result, expected), do: expected - result

  def calculate_delta_output(sum, error) do
    sigmoid_prime(sum) * error
  end

end
