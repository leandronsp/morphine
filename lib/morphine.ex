defmodule Morphine do
  use Application

  @moduledoc """
  Documentation for Morphine.
  """

  def start(_type, _args) do
    Morphine.Supervisor.start_link
  end
end
