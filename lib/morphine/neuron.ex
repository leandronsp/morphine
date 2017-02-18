defmodule Morphine.Neuron do
  defstruct inputs: [], weights: [], bias: 1

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

      iex> Morphine.Neuron.sigmoid_prime(1.235)
      0.17454403433618204

      iex> Morphine.Neuron.sigmoid_prime(1.0)
      0.19661193324148188

      iex> Morphine.Neuron.activation(0.5800000000000001, :sigmoid)
      0.6410674063348171

      iex> Morphine.Neuron.apply_bias(0.73, %Morphine.Neuron{bias: 0.3})
      0.219

      iex> Morphine.Neuron.error_margin(0.8, 1)
      0.19999999999999996
  """

  def summation(%{inputs: [], weights: []}), do: 0

  def summation(%{inputs: [h1|t1], weights: [h2|t2]}) do
    h1 * h2 + summation(%{inputs: t1, weights: t2})
  end

  def apply_bias(summation, neuron), do: summation * neuron.bias

  def sigmoid(calculation), do: 1 / (1 + :math.exp(-calculation))

  def sigmoid_prime(calculation) do
    :math.exp(-calculation) / :math.pow(1 + :math.exp(-calculation), 2)
  end

  def calculation(neuron) do
    summation(neuron)
    |> apply_bias(neuron)
  end

  def activation(calculation, :sigmoid) do
    calculation |> sigmoid
  end

  def error_margin(result, expected), do: expected - result

  def forward_propagation(neuron, target) do
    summation   = summation(neuron)
    summation_with_sigmoid = activation(summation, :sigmoid)
    calculation = summation_with_sigmoid * neuron.bias
    output      = activation(calculation, :sigmoid)
    error       = error_margin(output, target)

    {target, error, summation, summation_with_sigmoid, calculation}
  end

  def back_propagation({target, error, summation, summation_with_sigmoid, calculation}, neuron) do
    delta_a = sigmoid_prime(calculation) * error
    delta_b = (delta_a / neuron.bias) * sigmoid_prime(summation)

    new_bias    = neuron.bias + (delta_a / summation_with_sigmoid)
    new_weights = Enum.map(neuron.weights, &(&1 + delta_b))

    build_neuron(neuron.inputs, new_weights, new_bias)
  end

  def learn(neuron, target) do
    Morphine.Neuron.forward_propagation(neuron, target)
    |> Morphine.Neuron.back_propagation(neuron)
  end

  def learn!(neuron, target) do
    predict(neuron) |> learn!(neuron, target)
  end

  def learn!(result, neuron, target) do
    smarter = learn(neuron, target)
    result  = predict(neuron)

    case result |> Float.round(2) == target do
      true  -> smarter
      false -> learn!(result, smarter, target)
    end
  end

  def predict(neuron) do
    summation(neuron)
    |> activation(:sigmoid)
    |> apply_bias(neuron)
    |> activation(:sigmoid)
    |> Float.round(2)
  end

  defp build_neuron(inputs, weights, bias) do
    %Morphine.Neuron{inputs: inputs, weights: weights, bias: bias}
  end

end
