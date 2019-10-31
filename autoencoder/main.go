package main

import (
	"flag"
	"fmt"
	"image"
	"image/png"
	"log"
	"math"
	"math/rand"
	"os"

	_ "net/http/pprof"

	gg "gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/examples/mnist"
	"gorgonia.org/tensor"

	"time"

	"gopkg.in/cheggaaa/pb.v1"
)

var (
	epochs     = flag.Int("epochs", 2, "Number of epochs to train for")
	dataset    = flag.String("dataset", "train", "Which dataset to train on? Valid options are \"train\" or \"test\"")
	dtype      = flag.String("dtype", "float64", "Which dtype to use")
	batchsize  = flag.Int("batchsize", 100, "Batch size")
	cpuprofile = flag.String("cpuprofile", "", "CPU profiling")
)

const mnistPath = "./mnist/"
const imgPath = "./images/"

var dt tensor.Dtype

func parseDtype() {
	switch *dtype {
	case "float64":
		dt = tensor.Float64
	case "float32":
		dt = tensor.Float32
	default:
		log.Fatalf("Unknown dtype: %v", *dtype)
	}
}

type nn struct {
	g              *gg.ExprGraph
	w1, w2, w3, w4 *gg.Node
	b1, b2, b3, b4 *gg.Node
	mOnes          *gg.Node

	out     *gg.Node
	predVal gg.Value
}

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

func newNN(g *gg.ExprGraph) *nn {
	// Create node for w/weight
	w1 := gg.NewMatrix(g, dt, gg.WithShape(784, 250), gg.WithName("w1"), gg.WithInit(gg.GlorotN(1.0)))
	w2 := gg.NewMatrix(g, dt, gg.WithShape(250, 50), gg.WithName("w2"), gg.WithInit(gg.GlorotN(1.0)))
	w3 := gg.NewMatrix(g, dt, gg.WithShape(50, 250), gg.WithName("w3"), gg.WithInit(gg.GlorotN(1.0)))
	w4 := gg.NewMatrix(g, dt, gg.WithShape(250, 784), gg.WithName("w4"), gg.WithInit(gg.GlorotN(1.0)))

	b1 := gg.NewMatrix(g, dt, gg.WithShape(1, 250), gg.WithName("b1"), gg.WithInit(gg.GlorotN(1.0)))
	b2 := gg.NewMatrix(g, dt, gg.WithShape(1, 50), gg.WithName("b2"), gg.WithInit(gg.GlorotN(1.0)))
	b3 := gg.NewMatrix(g, dt, gg.WithShape(1, 250), gg.WithName("b3"), gg.WithInit(gg.GlorotN(1.0)))
	b4 := gg.NewMatrix(g, dt, gg.WithShape(1, 784), gg.WithName("b4"), gg.WithInit(gg.GlorotN(1.0)))

	// matrix of ones
	mOnes := gg.NewMatrix(g, dt, gg.WithShape(*batchsize, 1), gg.WithName("mOnes"), gg.WithInit(gg.Ones()))

	return &nn{
		g:     g,
		w1:    w1,
		w2:    w2,
		w3:    w3,
		w4:    w4,
		b1:    b1,
		b2:    b2,
		b3:    b3,
		b4:    b4,
		mOnes: mOnes,
	}
}

func (m *nn) learnables() gg.Nodes {
	return gg.Nodes{m.w1, m.w2, m.w3, m.w4, m.b1, m.b2, m.b3, m.b4}
}

func (m *nn) fwd(x *gg.Node) (err error) {
	var l0, l1, l2, l3, l4 *gg.Node

	// Set first layer to be copy of input
	l0 = x

	// gg.Must suppresses the err
	// gg will soon get an update that will make this unnecessary
	l1 = gg.Must(gg.Rectify(
		gg.Must(gg.Add(
			gg.Must(gg.Mul(l0, m.w1)),
			gg.Must(gg.Mul(m.mOnes, m.b1)),
		)),
	))

	l2 = gg.Must(gg.Rectify(
		gg.Must(gg.Add(
			gg.Must(gg.Mul(l1, m.w2)),
			gg.Must(gg.Mul(m.mOnes, m.b2)),
		)),
	))

	l3 = gg.Must(gg.Rectify(
		gg.Must(gg.Add(
			gg.Must(gg.Mul(l2, m.w3)),
			gg.Must(gg.Mul(m.mOnes, m.b3)),
		)),
	))

	l4 = gg.Must(gg.Sigmoid(
		gg.Must(gg.Add(
			gg.Must(gg.Mul(l3, m.w4)),
			gg.Must(gg.Mul(m.mOnes, m.b4)),
		)),
	))

	// set out output to the last layer
	m.out = l4
	gg.Read(m.out, &m.predVal)
	return

}

const pixelRange = 255

func reversePixelWeight(px float64) byte {
	// ensure our output is bounded within our known pixel range
	return byte(pixelRange*math.Min(0.99, math.Max(0.01, px)) - pixelRange)
}

func visualizeRow(x []float64) *image.Gray {
	// since we know MNIST is a square, we can take advantage of that
	l := len(x)
	side := int(math.Sqrt(float64(l)))
	r := image.Rect(0, 0, side, side)
	img := image.NewGray(r)

	pix := make([]byte, l)
	for i, px := range x {
		pix[i] = reversePixelWeight(px)
	}
	img.Pix = pix

	return img
}

func main() {
	flag.Parse()
	parseDtype()
	rand.Seed(7945)

	var inputs tensor.Tensor
	var err error

	// load our data set
	trainOn := *dataset
	if inputs, _, err = mnist.Load(trainOn, mnistPath, dt); err != nil {
		log.Fatal(err)
	}

	numExamples := inputs.Shape()[0]
	bs := *batchsize

	g := gg.NewGraph()
	x := gg.NewMatrix(g, dt, gg.WithShape(bs, 784), gg.WithName("x"))
	y := gg.NewMatrix(g, dt, gg.WithShape(bs, 784), gg.WithName("y"))

	m := newNN(g)
	if err = m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	cost := gg.Must(gg.Mean(
		gg.Must(gg.Square(
			gg.Must(gg.Sub(y, m.out)),
		)),
	))

	// track costs!
	var costVal gg.Value
	gg.Read(cost, &costVal)

	if _, err = gg.Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	vm := gg.NewTapeMachine(g, gg.BindDualValues(m.learnables()...))
	solver := gg.NewRMSPropSolver(gg.WithBatchSize(float64(bs)))

	batches := numExamples / bs
	log.Printf("Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second / 20)
	bar.SetMaxWidth(80)

	for i := 0; i < *epochs; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			var xVal tensor.Tensor
			if xVal, err = inputs.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice x")
			}

			if err = xVal.(*tensor.Dense).Reshape(bs, 784); err != nil {
				log.Fatalf("Unable to reshape %v", err)
			}

			gg.Let(x, xVal)
			gg.Let(y, xVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d: %v", i, err)
			}

			solver.Step(gg.NodesToValueGrads(m.learnables()))
			vm.Reset()
			bar.Increment()
		}
		bar.Update()
		log.Printf("Epoch %d | cost %v", i, costVal)
	}
	bar.Finish()

	log.Printf("Run Tests")

	// load our test set
	if inputs, _, err = mnist.Load("test", mnistPath, dt); err != nil {
		log.Fatal(err)
	}

	// prep images directory if it is missing
	if _, err := os.Stat(imgPath); os.IsNotExist(err) {
		os.Mkdir(imgPath, os.ModeDir)
	}

	numExamples = inputs.Shape()[0]
	bs = *batchsize
	batches = numExamples / bs

	bar = pb.New(batches)
	bar.SetRefreshRate(time.Second / 20)
	bar.SetMaxWidth(80)
	bar.Prefix(fmt.Sprintf("Epoch Test"))
	bar.Set(0)

	bar.Start()
	for b := 0; b < batches; b++ {
		start := b * bs
		end := start + bs
		if start >= numExamples {
			break
		}
		if end > numExamples {
			end = numExamples
		}

		var xVal tensor.Tensor
		if xVal, err = inputs.Slice(sli{start, end}); err != nil {
			log.Fatal("Unable to slice x")
		}

		if err = xVal.(*tensor.Dense).Reshape(bs, 784); err != nil {
			log.Fatalf("Unable to reshape %v", err)
		}

		gg.Let(x, xVal)
		gg.Let(y, xVal)
		if err = vm.RunAll(); err != nil {
			log.Fatalf("Failed at epoch test: %v", err)
		}

		// write the input image into a PNG
		for j := 0; j < xVal.Shape()[0]; j++ {
			rowT, _ := xVal.Slice(sli{j, j + 1})
			row := rowT.Data().([]float64)

			img := visualizeRow(row)

			f, _ := os.OpenFile(fmt.Sprintf("%v%d - %d input.png", imgPath, b, j), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
			png.Encode(f, img)
			f.Close()
		}

		// write the output image into a PNG
		arrayOutput := m.predVal.Data().([]float64)
		yOutput := tensor.New(tensor.WithShape(bs, 784), tensor.WithBacking(arrayOutput))

		for j := 0; j < yOutput.Shape()[0]; j++ {
			rowT, _ := yOutput.Slice(sli{j, j + 1})
			row := rowT.Data().([]float64)

			img := visualizeRow(row)

			f, _ := os.OpenFile(fmt.Sprintf("%v%d - %d output.png", imgPath, b, j), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
			png.Encode(f, img)
			f.Close()
		}

		vm.Reset()
		bar.Increment()
	}
	log.Printf("Epoch Test | cost %v", costVal)
}
