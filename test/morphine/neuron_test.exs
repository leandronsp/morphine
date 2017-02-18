defmodule Morphine.NeuronTest do
  use ExUnit.Case, async: true
  doctest Morphine.Neuron

  test "forward propagation" do
    neuron = %Morphine.Neuron{inputs: [1, 1], weights: [0.8, 0.2], bias: 0.3}

    {target, error, summation, summation_with_sigmoid, calculation} =
    Morphine.Neuron.forward_propagation(neuron, 0)

    assert target == 0
    assert error  == -0.5546106700105877
    assert summation == 1.0
    assert summation_with_sigmoid == 0.7310585786300049
    assert calculation == 0.21931757358900147
  end

  test "back propagation" do
    neuron = %Morphine.Neuron{inputs: [1, 1], weights: [0.8, 0.2], bias: 0.3}

    new_neuron =
    Morphine.Neuron.forward_propagation(neuron, 0)
    |> Morphine.Neuron.back_propagation(neuron)

    assert new_neuron.bias == 0.11260237949987248
    assert Enum.at(new_neuron.weights, 0) == 0.7102147763845081
    assert Enum.at(new_neuron.weights, 1) == 0.11021477638450808
  end

  test "learn" do
    neuron =
    %Morphine.Neuron{inputs: [1, 1], weights: [0.8, 0.2], bias: 0.3}
    |> Morphine.Neuron.learn(0)

    assert neuron.bias == 0.11260237949987248
    assert Enum.at(neuron.weights, 0) == 0.7102147763845081
    assert Enum.at(neuron.weights, 1) == 0.11021477638450808
  end

  test "predict" do
    neuron = %Morphine.Neuron{inputs: [1, 1], weights: [0.8, 0.2], bias: 0.3}
    assert Morphine.Neuron.predict(neuron) == 0.55
  end

  test "learn N times then predict" do
    neuron =
    %Morphine.Neuron{inputs: [1, 1], weights: [0.8, 0.2], bias: 0.3}
    |> Morphine.Neuron.learn(0)
    |> Morphine.Neuron.learn(0)
    |> Morphine.Neuron.learn(0)

    assert Morphine.Neuron.predict(neuron) == 0.45
  end

  test "learn! until it reaches target" do
    neuron = %Morphine.Neuron{inputs: [1, 1], weights: [0.8, 0.2], bias: 0.3}
    smarter = Morphine.Neuron.learn!(neuron, 0)

    assert Morphine.Neuron.predict(smarter) == 0
  end

end
