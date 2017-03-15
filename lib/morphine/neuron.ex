defmodule Morphine.Neuron do
  @moduledoc """
  A simple Neuron, with weights and bias.
  """

  defstruct weights: [], bias: 1

  @type neuron :: Morphine.Neuron

  @doc """
  Builds a neuron given weights.

  ## Examples

      iex>Morphine.Neuron.build([0.33, 0.42])
      %Morphine.Neuron{weights: [0.33, 0.42]}
  """

  @spec build(list) :: neuron
  def build(weights), do: %Morphine.Neuron{weights: weights}

end
