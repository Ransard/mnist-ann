package neural

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"strconv"
)

type Network struct {
	WHI [][]float64
	WOH [][]float64
}

var learningRate = 0.005

func (n *Network) backpropagate(answer int, outputValues []float64, hiddenValues []float64, inputValues []string) {

	var outputErrors []float64

	for ix, _ := range outputValues {
		var normalizedAnswer float64
		if ix == answer {
			normalizedAnswer = 1.0
		} else {
			normalizedAnswer = 0.0
		}

		outputErrors = append(outputErrors, normalizedAnswer-outputValues[ix])
	}

	hiddenErrors := make([]float64, len(n.WOH[0]))

	for io, _ := range n.WOH {

		for ih, weight := range n.WOH[io] {
			hiddenErrors[ih] += outputErrors[io] * weight
		}
	}

	for io, _ := range n.WOH {
		for ih, _ := range n.WOH[io] {
			n.WOH[io][ih] += learningRate * outputErrors[io] * outputValues[io] * (1.0 - outputValues[io]) * hiddenValues[ih]
		}
	}

	for ih, _ := range n.WHI {
		for ii, _ := range n.WHI[ih] {
			inputVal, _ := strconv.ParseFloat(inputValues[ii], 64)
			n.WHI[ih][ii] += learningRate * hiddenErrors[ih] * hiddenValues[ih] * (1.0 - hiddenValues[ih]) * inputVal
		}
	}
}

func (n *Network) Train(input [][]string) int {
	wins := 0
	for _, data := range input {
		inputData := data[1:]
		output, hidden, _ := n.Query(inputData)
		answer, _ := strconv.Atoi(data[0])

		if GetResult(output) == answer {
			wins++
		}

		n.backpropagate(answer, output, hidden, inputData)
	}

	return wins * 100 / len(input)
}

func GetResult(result []float64) int {
	output := -1.0
	index := -1

	for i, value := range result {
		if value > output {
			index = i
			output = value
		}
	}
	return index
}

func (n *Network) Query(input []string) ([]float64, []float64, error) {

	if len(input) > len(n.WHI[0]) {
		return nil, nil, errors.New("nope")
	}

	hiddenValues := make([]float64, 0)

	for ih, _ := range n.WHI {
		sum := 0.0
		for ii, weight := range n.WHI[ih] {
			inputValue, _ := strconv.ParseFloat(input[ii], 64)
			sum += inputValue * weight / 255.0 * 0.99
		}

		hiddenValues = append(hiddenValues, sigmoid(sum))
	}

	outputValues := make([]float64, 0)

	for io, _ := range n.WOH {
		sum := 0.0
		for ih, weight := range n.WOH[io] {
			sum += hiddenValues[ih] * weight
		}

		outputValues = append(outputValues, sigmoid(sum))
	}

	return outputValues, hiddenValues, nil
}

func sigmoid(value float64) float64 {
	return 1.0 / (1.0 + math.Exp(-value))
}

func (n *Network) Init(input, hidden, output int) {
	fmt.Println("Initializing network")

	fmt.Println("sigmoid func is ", sigmoid(10.0))

	whi := make([][]float64, hidden)
	for i := range whi {
		whi[i] = make([]float64, input)

		for j := range whi[i] {
			whi[i][j] = rand.Float64() - 0.5
		}
	}

	n.WHI = whi

	woh := make([][]float64, output)
	for i := range woh {
		woh[i] = make([]float64, hidden)

		for j := range woh[i] {
			woh[i][j] = rand.Float64() - 0.5
		}
	}

	n.WOH = woh
}
