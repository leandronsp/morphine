defmodule Morphine.NeuralNetworkTest do
  use ExUnit.Case, async: true
  doctest Morphine.NeuralNetwork

  alias Morphine.Layer
  alias Morphine.Neuron
  alias Morphine.NeuralNetwork, as: Network
  alias Morphine.Calc.Matrix, as: Matrix

  setup do
    {:ok, network} = Network.start_link
    {:ok, network: network}
  end

  test "#learn and #predict given specific weights", %{network: network} do
    neuron_1 = %Neuron{weights: [-0.16595599, -0.70648822, -0.20646505]}
    neuron_2 = %Neuron{weights: [0.44064899, -0.81532281, 0.07763347]}
    neuron_3 = %Neuron{weights: [-0.99977125, -0.62747958, -0.16161097]}
    neuron_4 = %Neuron{weights: [-0.39533485, -0.30887855, 0.370439]}
    neuron_5 = %Neuron{weights: [-0.5910955, 0.75623487, -0.94522481, 0.34093502]}
    layer_1  = %Layer{neurons: [neuron_1, neuron_2, neuron_3, neuron_4]}
    layer_2  = %Layer{neurons: [neuron_5]}

    Network.setup_layers(network, [{4, 3}, {1, 4}])
    Network.update_layers!(network, [layer_1, layer_2])

    ### XOR gate
    inputs = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
    outputs = Matrix.transpose([[0, 1, 1, 1, 1, 0, 0]])

    Network.learn(network, inputs, outputs, 60000)
    assert Network.predict!(network, [[1, 1, 0]]) == 0.007887604373626907
  end

end
