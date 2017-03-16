defmodule Morphine.BackPropagation do
  @moduledoc """
  Given the output, calculates the error, adjust and propagate across layers.
  """

  alias Morphine.Layer
  alias Morphine.Calc.Activation, as: Activation
  alias Morphine.Calc.Matrix, as: Matrix

  @type layer :: Morphine.Layer

  @doc """
  Calculates delta for the output layer.
  It takes the output of current layer, the output layer and the delta of output layer.

  ## Examples

    iex> Morphine.BackPropagation.delta([[1, 0, 1]], [[0.42, 0.31, 0.12]])
    [[0.14128800000000002, -0.06630899999999999, 0.092928]]
  """

  @spec delta(list, list) :: list
  def delta(target, output) do
    func  = fn array -> Enum.map(array, &Activation.sigmoid_derivative(&1)) end
    error = Matrix.subtract(target, output)
    Enum.map(output, &func.(&1)) |> Matrix.naive_multiply(error)
  end

  @doc """
  Calculates delta for any layer but the output one.
  It takes the target and the output of current layer.

  ## Examples

    iex> output_layer = Morphine.Layer.from_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    %Morphine.Layer{neurons: [
      %Morphine.Neuron{weights: [1, 4, 7]},
      %Morphine.Neuron{weights: [2, 5, 8]},
      %Morphine.Neuron{weights: [3, 6, 9]}
    ]}
    iex> Morphine.BackPropagation.delta([[9, 9, 9]], output_layer, [[4, 3, 8]])
    [[-2448, -5688, -8928]]
  """

  @spec delta(list, layer, list) :: list
  def delta(output, output_layer, delta_output_layer) do
    func  = fn array -> Enum.map(array, &Activation.sigmoid_derivative(&1)) end
    factor = Matrix.transpose(Layer.to_matrix(output_layer))
    error  = Matrix.multiply(delta_output_layer, factor)

    Enum.map(output, &func.(&1)) |> Matrix.naive_multiply(error)
  end

  @doc """
  ** UNIQUE LAYER **
  Adjusts layer using inputs.
  """
  @spec adjust(list, list, list) :: list
  def adjust(inputs, target, [{layer, output}]) do
    delta      = delta(target, output)
    adjusted   = apply_adjustment(inputs, delta, layer)

    [adjusted]
  end

  @doc """
  ** FIRST LAYER **
  Adjusts using inputs and the delta of last (output) layer.
  """
  def adjust(inputs, target, [{layer, output}|remaining]) do
    {last_layer, output_last_layer} = fetch_output_layer(remaining)

    deltanl    = delta(target, output_last_layer)
    delta      = delta(output, last_layer, deltanl)
    adjusted   = apply_adjustment(inputs, delta, layer)

    adjust(inputs, target, remaining, output, [adjusted])
  end

  @doc """
  ** MIDDLE LAYER **
  Adjusts using output of previous layer.
  """
  def adjust(_, target, [{layer, output}|[]], output_previous_layer, acc) do
    delta      = delta(target, output)
    adjusted   = apply_adjustment(output_previous_layer, delta, layer)

    acc ++ [adjusted]
  end

  @doc """
  ** MIDDLE LAYER **
  Adjusts using output of previous layer and the delta of last (output) layer.
  """
  def adjust(inputs, target, [{layer, output}|remaining], output_previous_layer, acc) do
    {last_layer, output_last_layer} = fetch_output_layer(remaining)

    deltanl    = delta(target, output_last_layer)
    delta      = delta(output, last_layer, deltanl)
    adjusted   = apply_adjustment(output_previous_layer, delta, layer)

    adjust(inputs, target, remaining, output, acc ++ [adjusted])
  end

  @spec fetch_output_layer(list) :: tuple
  defp fetch_output_layer(list) do
    {{last_layer, output_last_layer}, _} = List.pop_at(list, length(list) - 1)
    {last_layer, output_last_layer}
  end

  @spec apply_adjustment(list, list, layer) :: layer
  defp apply_adjustment(data, delta, layer) do
    adjustment = Matrix.transpose(data) |> Matrix.multiply(delta)

    Layer.to_matrix(layer)
    |> Matrix.add(adjustment)
    |> Layer.from_matrix
  end

end
