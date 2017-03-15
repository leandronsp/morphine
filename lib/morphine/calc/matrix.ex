defmodule Morphine.Calc.Matrix do
  @moduledoc """
  Matrix manipulations.
  """

  @doc """
  A simple multiplier for matrices.

  ## Examples

    iex> Morphine.Calc.Matrix.naive_multiply([[1, 2, 3]], [[2, 2, 2]])
    [[2, 4, 6]]
  """

  def naive_multiply([], []), do: []
  def naive_multiply([a_first_row | a_remaining_rows], [b_first_row | b_remaining_rows]) do
    [do_naive_multiply(a_first_row, b_first_row) | naive_multiply(a_remaining_rows, b_remaining_rows)]
  end

  defp do_naive_multiply([], []), do: []
  defp do_naive_multiply([u_head | u_tail], [v_head | v_tail]) do
    [u_head * v_head | do_naive_multiply(u_tail, v_tail)]
  end

  @doc """
  These operations below use a third-party lib for matrix manipulations.

  ## Examples

    iex> Morphine.Calc.Matrix.multiply([[1], [2], [3]], [[2, 2, 2]])
    [[2, 2, 2], [4, 4, 4], [6, 6, 6]]

    iex> Morphine.Calc.Matrix.add([[1, 2, 3]], [[1, 1, 1]])
    [[2, 3, 4]]

    iex> Morphine.Calc.Matrix.subtract([[2, 3, 4]], [[1, 1, 1]])
    [[1, 2, 3]]

    iex> Morphine.Calc.Matrix.transpose([[0, 1, 0, 1]])
    [[0], [1], [0], [1]]
  """

  def add(a, b),      do: ExAlgebra.Matrix.add(a, b)
  def multiply(a, b), do: ExAlgebra.Matrix.multiply(a, b)
  def subtract(a, b), do: ExAlgebra.Matrix.subtract(a, b)
  def transpose(a),   do: ExAlgebra.Matrix.transpose(a)

end
