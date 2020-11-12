# TOYDL : A Toy Deep Learning Framework

This is my assignment's solution in the course **AI6104: Mathematics for Artificial Intelligence** in NTU.

This framework is really naive. It use numpy as the backend and only implement several simple layers like *Linear*, *ReLU* and some losses like *MSE* and *CrossEntropyLoss*.

## Architecture

I tried to make the API more like PyTorch but emm... With limited time since I was kept busy on something else and the deadline, I cannot say that I did a good job.

The architecture is simple and straightforward. `tensor.py` implement the base tensor used through model, it's like *Caffe*'s blob, at least I think.

`Module` is the `class` that take response to `forward` and `backward`, again I think, it's like *PyTorch*'s `Module`.


## Demo

There is a demo which use a MLP to train MNIST and I upload the log file I got for reference.

To run the demo by yourself, just run following code

```shell
# First cd into the test directory
cd test

# Then just run it
python test_mlp.py
```

## Future work

Implementing your own deep learning framework is actully an excited thing. If I have more time in the future which i doubt there are several things I want to do:

* Define a graph that manage all the layers
* Define a engine that runs the graph which supports optimization
* Add more operators
