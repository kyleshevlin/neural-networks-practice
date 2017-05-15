// Let's teach a neural network XOR (exclusive or)
const { Layer, Network } = require('synaptic')

// XOR takes 2 inputs
// These inputs can either be true or false
const inputLayer = new Layer(2)

// There are 3 combinations from the two inputs
// 1. true, true
// 2. (true, false) || (false, true)
// 3. false, false
const hiddenLayer = new Layer(3)

// XOR has 1 output true or false
const outputLayer = new Layer(1)

// This describes the projection of layers in our network.
// Each neuron in the layer is projected onto each neuron in the next layer
inputLayer.project(hiddenLayer)
hiddenLayer.project(outputLayer)

// Setup our neural network
const xorNetwork = new Network({
  input: inputLayer,
  hidden: [hiddenLayer],
  output: outputLayer
})

// Train the network

/*
After each loop, the network will examine how close it came
to the getting the correct answer. It will then make an adjustment
to "weight" the next guess based on the learningRate
*/
const learningRate = 0.3

// The network learns by doing, a lot of times
for (let i = 0; i < 20000; i++) {
  // We will use 0s for false and 1s for true
  // as they will be coerced to Boolean values by the JS engine

  // (false, false) => false
  xorNetwork.activate([0, 0])
  xorNetwork.propagate(learningRate, [0])

  // (false, true) => true
  xorNetwork.activate([0, 1])
  xorNetwork.propagate(learningRate, [1])

  // (true, false) => true
  xorNetwork.activate([1, 0])
  xorNetwork.propagate(learningRate, [1])

  // (true, true) => false
  xorNetwork.activate([1, 1])
  xorNetwork.propagate(learningRate, [0])
}

// Test how well the xorNetwork learned
console.log('0, 0', xorNetwork.activate([0, 0]))
console.log('0, 1', xorNetwork.activate([0, 1]))
console.log('1, 0', xorNetwork.activate([1, 0]))
console.log('1, 1', xorNetwork.activate([1, 1]))
