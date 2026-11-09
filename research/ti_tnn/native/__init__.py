"""Native, randomly initialized TI-TNN model bring-up and its shared GPU accounting.

For example, 1 hour of native compilation and profiling after the 23.951425708866115 historical hours leaves 11 of
the 12 additional hours. Loading, warm-up, compilation, profiling, training, evaluation, failures, and GPU
reproduction all count. Authorization v3 permits that engineering work; it is not evidence that the model learns.

The CPU ledger tests check construction and accounting with temporary files. Learning evidence requires separate
training and evaluation results, and neither these tests nor a reservation establishes it.
"""
