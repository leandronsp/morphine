defmodule Morphine.Layer do
  alias Morphine.Neuron
  defstruct neurons: []

  @moduledoc """
  Documentation for Morphine.Layer.
  """

  @doc """
  A simple Layer.

  ## Examples

      iex> layer = Morphine.Layer.build(4, 3, :random)
      iex> length(layer.neurons)
      4

      iex> Morphine.Layer.build(1, 4)
      %Morphine.Layer{neurons: [%Morphine.Neuron{weights: [1, 1, 1, 1]}]}

      iex> layer = Morphine.Layer.build(1, 4)
      iex> Morphine.Layer.to_matrix(layer)
      [[1], [1], [1], [1]]

      iex> layer = Morphine.Layer.build(4, 3)
      iex> Morphine.Layer.to_matrix(layer)
      [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

      iex> layer = Morphine.Layer.build(3, 4)
      iex> Morphine.Layer.to_matrix(layer)
      [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]

      iex> Morphine.Layer.from_matrix([[1], [2], [3], [4]])
      %Morphine.Layer{neurons: [%Morphine.Neuron{weights: [1, 2, 3, 4]}]}

      iex> Morphine.Layer.from_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      %Morphine.Layer{neurons: [
        %Morphine.Neuron{weights: [1, 4, 7]},
        %Morphine.Neuron{weights: [2, 5, 8]},
        %Morphine.Neuron{weights: [3, 6, 9]}
      ]}

  """

  def build(number_of_neurons, number_of_weights) do
    neurons = for _ <- 1..number_of_neurons do
      weights = for _ <- 1..number_of_weights, do: 1
      Neuron.build(weights)
    end

    %Morphine.Layer{neurons: neurons}
  end

  def build(number_of_neurons, number_of_weights, :random) do
    neurons = for _ <- 1..number_of_neurons do
      weights = for _ <- 1..number_of_weights, do: 2 * :rand.uniform - 1
      Neuron.build(weights)
    end

    %Morphine.Layer{neurons: neurons}
  end

  def to_matrix(layer) do
    Enum.map(layer.neurons, fn neuron -> neuron.weights end)
    |> ExAlgebra.Matrix.transpose
  end

  def from_matrix(matrix) do
    tmatrix = ExAlgebra.Matrix.transpose(matrix)
    neurons = Enum.map(tmatrix, &Neuron.build/1)

    %Morphine.Layer{neurons: neurons}
  end
end
