import sys
import os
import shutil
import zipfile


def copyMods():
    dirs = ['path1', 'path2', 'path3']

    # Clean out any old package paths
    for d in dirs:
        if os.path.isdir(d):
            shutil.rmtree(d)

    for d in dirs:
        os.mkdir(d)
        os.mkdir(os.path.join(d, 'brave'))

    shutil.copy('robin.py', os.path.join('path1', 'brave'))
    os.system('cp _robin.* ' + os.path.join('path1', 'brave'))

    shutil.copy('robin.py', os.path.join('path2', 'brave'))
    os.system('cp _robin.* ' + os.path.join('path3', 'brave'))

    mkzip()

def mkzip():
    zf = zipfile.ZipFile("path4.zip", "w")
    zf.writestr("brave/", b'')
    zf.write('robin.py', 'brave/robin.py')
    zf.close()


def main():
    copyMods()

    # Run each test with a separate interpreter
    os.system(sys.executable + " nonpkg.py")
    os.system(sys.executable + " normal.py")
    os.system(sys.executable + " split.py")
    os.system(sys.executable + " zipsplit.py")


if __name__ == "__main__":
    main()
