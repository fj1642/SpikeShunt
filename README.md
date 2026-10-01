# SpikeShunt SI
SpikeShunt-SI, in this we shunt snn into Ann structures, to reduce time and delay, how to take in a structure of ann, in which form and through which algorithm.
Hybrid ANN/SNN builder.

Takes a plain ANN (a stack of Linear+ReLU blocks), profiles each block two
ways -- as dense ANN and as a rate-coded SNN with the SAME weights -- and
swaps in the SNN version only where it actually measures cheaper, under
whichever cost metric you pick:

  - latency       : real wall-clock forward time. This is the literal
                       answer to "would it run faster right now, on this
                       machine". Be warned: on ordinary dense CPU/GPU
                       tensors this almost never favors the SNN -- a
                       T-step loop over the same matmul costs roughly T x
                       more than one dense pass, no matter how sparse the
                       activations are, *unless the sparsity is actually
                       exploited in the compute*. Vanilla PyTorch tensor
                       ops don't do that automatically.

  - effective_ops  : estimated multiply-accumulate count, using each
                       block's measured spike density to count only the
                       work a sparse / event-driven implementation (real
                       neuromorphic hardware, or a proper sparse kernel)
                       would actually pay. This is the metric that matches
                       *why* people expect SNNs to win -- it is a fair
                       proxy for that kind of backend, not for naive dense
                       PyTorch ops on a CPU/GPU.

Both are implemented and both are printed, so the comparison is honest
instead of quietly picking whichever metric makes the demo "work".
"""
