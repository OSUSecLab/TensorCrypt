## File Structure
`aes_cipher.py`: inference script for AES cipher
`chacha20_cipher.py`: inference script for Chacha
`salsa20_cipher.py`: inference script for Salsa

## Notes

### Hardware Diversity

Our evaluations are
performed on diverse platforms listed in Table I in [our paper](https://www.ndss-symposium.org/wp-content/uploads/2025-955-paper.pdf). While there
can be other platforms for deployment and evaluations, the
performance observed on these platforms may differ from
the results reported in this paper. We believe the evaluations
on other platforms (e.g., other GPUs) will be interesting
for future research. Additionally, we selected the baseline
implementations to the best of our ability; however, more
advanced baseline implementations may emerge in the future.
For TENSORCRYPT models, we only implemented them on
TensorFlow, and we believe there are frameworks for implementations. While our primary contributions focus on the
novel application of neural networks (NN) for cryptography,
we leave the exploration of alternative implementations as
future work.

### Enable XLA

Make sure XLA has been enabled during inference time, e.g., you probably can see logging messages, like:
```plaintext
I0000 00:00:1743315015.877756   99410 service.cc:152] XLA service 0x368500a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1743315015.877781   99410 service.cc:160]   StreamExecutor device (0): Host, Default Version
I0000 00:00:1743315016.170292   99410 device_compiler.h:188] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
```