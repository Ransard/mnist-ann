package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"

	"strconv"

	. "bitbucket.org/Cabe/ann-mnist/neural"
)

func main() {
	fmt.Println("Hello!")
	fmt.Println("Lets get started learning some MNIST data, eh?")
	myNetwork := Network{}
	myNetwork.Init(784, 200, 10)
	train := readData("data/mnist_train.csv", -1)
	test := readData("data/mnist_test.csv", -1)

	fmt.Println("Starting training")

	myNetwork.Train(train)

	fmt.Println("Starting testing")

	wins := 0
	for _, data := range test {
		result, _, err := myNetwork.Query(data[1:])

		if err != nil {
			fmt.Println("error is ", err)
		} else {
			index := GetResult(result)

			//fmt.Println("result is ", index, "answer is ", test[0][0])
			answer, _ := strconv.Atoi(data[0])
			if index == answer {
				wins++
			}

		}
	}

	fmt.Println("perc is ", wins*100/len(test))

}

func readData(file string, maxRead int) [][]string {
	result := [][]string{}
	f, _ := os.Open(file)
	r := csv.NewReader(bufio.NewReader(f))

	for {
		record, err := r.Read()

		if err == io.EOF || (len(result) >= maxRead && maxRead != -1) {
			break
		}

		result = append(result, record)
	}

	return result
}
