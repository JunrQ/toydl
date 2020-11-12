# TOYDL : A Toy Deep Learning Framework

This is my assignment's solution in the course **AI6104: Mathematics for Artificial Intelligence** in NTU.

This framework is really naive. It use numpy as the backend and only implement several simple layers like *Linear*, *ReLU* and some losses like *MSE* and *CrossEntropyLoss*.

## Architecture

I tried to make the API more like PyTorch but emm... With limited time since I was kept busy on something else and the deadline, I cannot say that I did a good job.

The architecture is simple and straightforward. `tensor.py` implement the base tensor used through model, it's like *Caffe*'s blob, at least I think.

`Module` is the `class` that take response to `forward` and `backward`, again I think, it's like *PyTorch*'s `Module`.


## Demo

There is a demo which use a MLP to train MNIST and I upload the log file I got for reference.
