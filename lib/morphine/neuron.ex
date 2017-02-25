defmodule Morphine.Neuron do
  defstruct weights: []

  @moduledoc """
  Documentation for Morphine.Neuron.
  """

  @doc """
  A simple Neuron.

  ## Examples

      iex>Morphine.Neuron.build([0.33, 0.42])
      %Morphine.Neuron{weights: [0.33, 0.42]}
  """

  def build(weights), do: %Morphine.Neuron{weights: weights}

end
