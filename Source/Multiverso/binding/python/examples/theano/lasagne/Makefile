.PHONY: 2P1G1M0.1LR128B1S 4P1G1M0.1LR128B0S 4P1G1M0.1LR128B1S 4P1G1M0.05LR64B0S 4P1G1M0.05LR64B1S 8P1G1M0.1LR128B0S 8P1G1M0.1LR128B1S 8P1G1M0.05LR64B0S 8P1G1M0.05LR64B1S

1P1G1M0.1LR128B1S:
	mpirun -np 1 python ./Deep_Residual_Learning_CIFAR-10.py -lr 0.1 -b 128 -s True

4P1G1M0.1LR128B0S:
	mpirun -np 4 python ./Deep_Residual_Learning_CIFAR-10.py -lr 0.1 -b 128 -s False
4P1G1M0.1LR128B1S:
	mpirun -np 4 python ./Deep_Residual_Learning_CIFAR-10.py -lr 0.1 -b 128 -s True

4P1G1M0.05LR64B0S:
	mpirun -np 4 python ./Deep_Residual_Learning_CIFAR-10.py -lr 0.05 -b 64 -s False
4P1G1M0.05LR64B1S:
	mpirun -np 4 python ./Deep_Residual_Learning_CIFAR-10.py -lr 0.05 -b 64 -s True

8P1G1M0.1LR128B0S:
	mpirun -np 8 python ./Deep_Residual_Learning_CIFAR-10.py -lr 0.1 -b 128 -s False
8P1G1M0.1LR128B1S:
	mpirun -np 8 python ./Deep_Residual_Learning_CIFAR-10.py -lr 0.1 -b 128 -s True

8P1G1M0.05LR64B0S:
	mpirun -np 8 python ./Deep_Residual_Learning_CIFAR-10.py -lr 0.05 -b 64 -s False
8P1G1M0.05LR64B1S:
	mpirun -np 8 python ./Deep_Residual_Learning_CIFAR-10.py -lr 0.05 -b 64 -s True

