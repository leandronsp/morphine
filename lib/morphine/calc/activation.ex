defmodule Morphine.Calc.Activation do

  @doc """
  A sigmoid function.

  ## Examples

    iex> Morphine.Calc.Activation.sigmoid(0.42)
    0.6034832498647263
  """

  def sigmoid(calc), do: 1 / (1 + :math.exp(-calc))

  @doc """
  A sigmoid derivative function.

  ## Examples

    iex> Morphine.Calc.Activation.sigmoid_derivative(0.85)
    0.1275
  """

  def sigmoid_derivative(calc), do: calc * (1 - calc)
end
