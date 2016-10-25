if __name__ == '__main__':
    import time
    import subprocess

    start_time = time.time()
    subprocess.call("python CifarResNet_Distributed.py", stderr=subprocess.STDOUT)
    print("\n--- Non-distributed: %s seconds ---\n" % (time.time() - start_time))

    start_time = time.time()
    subprocess.call("mpiexec -n 2 python CifarResNet_Distributed.py", stderr=subprocess.STDOUT)
    print("\n--- 2 workers      : %s seconds ---\n" % (time.time() - start_time))
