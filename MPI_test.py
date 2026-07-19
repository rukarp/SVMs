from mpi4py import MPI
import socket

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = socket.gethostname()

# MPIのrankとhostを表示する
for i in range(size):
    if rank == i:
        print(f"rank={rank}, host={host}")
    comm.Barrier()