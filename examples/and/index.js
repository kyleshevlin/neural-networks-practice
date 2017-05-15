// Let's teach a neural network AND
const { Layer, Network } = require('synaptic')

const inputLayer = new Layer(2)
const hiddenLayer = new Layer(3)
const outputLayer = new Layer(1)

inputLayer.project(hiddenLayer)
hiddenLayer.project(outputLayer)

const andNetwork = new Network({
  input: inputLayer,
  hidden: [hiddenLayer],
  output: outputLayer
})

const learningRate = 0.3
const activate = argsArray => andNetwork.activate(argsArray)
const propagate = expected => andNetwork.propagate(learningRate, expected)

for (let i = 0; i < 20000; i++) {
  activate([0, 0])
  propagate([0])

  activate([0, 1])
  propagate([0])

  activate([1, 0])
  propagate([0])

  activate([1, 1])
  propagate([1])
}

console.log('0, 0', activate([0, 0]))
console.log('0, 1', activate([0, 1]))
console.log('1, 0', activate([1, 0]))
console.log('1, 1', activate([1, 1]))
