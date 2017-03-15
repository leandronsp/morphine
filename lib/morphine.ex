defmodule Morphine do
  use Application

  def start(_type, _args) do
    Morphine.Supervisor.start_link
  end
end
