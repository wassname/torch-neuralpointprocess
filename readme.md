Modified by wassname from the below:

Changes:
- [x] clamp weight with epsilon for stablity
  - from `p.data *= (p.data>=0)`
  - to `p.data = torch.clamp(p.data, min=eps)`
- [ ] try log t
- [x] try not mean as much in intensity layer
- [x] use pytorch lightning
- [x] fix potential data leak where label is fed to lsm
- [x] commit notebook with plots of val


# Fully Neural Network based Model for General Temporal Point Process(Neurips 2019,Takahiro Omi)

This code is pytorch version of implementation for Neural Temporal Point Process.

Temporal Point Process is mathematical model for capturing patterns of discrete event occurrences.
  However, the traditional point process have limited expressivity by assumption.
  For example, Poisson process assumes the independence of all events though it changes by time.
  Other point process like Hawkes process does not assume the independence but the intensity function
  of Hawkes process should be positive and have exponential decaying kernel. For these reason, the author
  suggest the generalized version of point process by introducing neural network.  


Reference

github : https://github.com/omitakahiro/NeuralNetworkPointProcess 



