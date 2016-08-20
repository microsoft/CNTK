// CrossProcessMutex.h -- implements a system-wide mutex to allow for system-wide GPU locking

#pragma once

// implementations differ greatly between Windows and Linux

#include <cassert>
#include <string>

#ifdef WIN32 // --- Windows version

#define NOMINMAX
#include "Windows.h" // for HANDLE

class CrossProcessMutex
{
    // no-copying
    CrossProcessMutex(const CrossProcessMutex&);
    void operator=(const CrossProcessMutex&);

    std::string m_name; // lock name
    HANDLE m_handle;

public:
    CrossProcessMutex(const std::string& name)
        : m_handle(NULL),
          m_name("Global\\" + name)
    {
    }

    // Acquires the mutex. If 'wait' is true and mutex is acquired by someone else then
    // function waits until mutex is released
    // Returns false if !wait and lock cannot be acquired, or in case of a system error that prevents us from acquiring the lock.
    bool Acquire(bool wait)
    {
        assert(m_handle == NULL);
        m_handle = ::CreateMutexA(NULL /*security attr*/, FALSE /*bInitialOwner*/, m_name.c_str());
        if (m_handle == NULL)
        {
            if (!wait)
                return false;   // can't lock due to access permissions: lock already exists, consider not available
            else
                RuntimeError("Acquire: Failed to create named mutex %s: %d.", m_name.c_str(), GetLastError());
        }

        if (::WaitForSingleObject(m_handle, wait ? INFINITE : 0) != WAIT_OBJECT_0)
        {
            // failed to acquire
            int rc = ::CloseHandle(m_handle);
            if ((rc == 0) && !std::uncaught_exception())
            {
                RuntimeError("Handler close failure with error code %d", ::GetLastError());
            }
            m_handle = NULL;
            return false;
        }

        return true;   // succeeded
    }

    // Releases the mutex
    void Release()
    {
        assert(m_handle != NULL);
        int rc = 0;
        rc = ::ReleaseMutex(m_handle);
        if ((rc == 0) && !std::uncaught_exception())
        {
            RuntimeError("Release: Failed to release mutex %s: %d", m_name.c_str(), ::GetLastError());
        }
        rc = ::CloseHandle(m_handle);
        if ((rc == 0) && !std::uncaught_exception())
        {
            RuntimeError("Release: Failed to close handle %s: %d", m_name.c_str(), ::GetLastError());
        }
        m_handle = NULL;
    }

    ~CrossProcessMutex()
    {
        if (m_handle != NULL)
        {
            Release();
        }
    }
};

#else // --- Linux version

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>

class CrossProcessMutex
{
    // no-copying
    CrossProcessMutex(const CrossProcessMutex&);
    void operator=(const CrossProcessMutex&);

    int m_fd;               // file descriptor
    std::string m_fileName; // lock file name
    struct flock m_lock;    // fnctl lock structure

    static void noOpAlarmHandler(int /*signum*/)
    {
        // this handler is intentionally NO-OP
        // the side effect of execution this handler
        // will be a termination of fcntl call below with EINTR
    }

    static void setupTimeout(int seconds)
    {
        struct sigaction action = {};
        action.sa_handler = &CrossProcessMutex::noOpAlarmHandler;
        sigaction(SIGALRM, &action, NULL);
        alarm(seconds);
    }

public:
    CrossProcessMutex(const std::string& name)
        : m_fd(-1),
          m_fileName("/var/lock/" + name)
    {
    }

    // Acquires the mutex. If 'wait' is true and mutex is acquired by someone else then
    // function waits until mutex is released
    // Returns false if !wait and lock cannot be acquired, or in case of a system error that prevents us from acquiring the lock.
    bool Acquire(bool wait)
    {
        assert(m_fd == -1);
        for (;;)
        {
            // opening a lock file
            int fd = open(m_fileName.c_str(), O_WRONLY | O_CREAT, 0666);
            if (fd < 0)
                RuntimeError("Acquire: Failed to open lock file %s: %s.", m_fileName.c_str(), strerror(errno));
            // locking it with the fcntl API
            memset(&m_lock, 0, sizeof(m_lock));
            m_lock.l_type = F_WRLCK;
            // BUG: fcntl call with F_SETLKW doesn't always reliably detect when lock is released
            // As a workaround, using alarm() for interupting fcntl if it waits more than 1 second
            setupTimeout(1);
            int r = fcntl(fd, wait ? F_SETLKW : F_SETLK, &m_lock);
            if (errno == EINTR)
            {
                sleep(1);
                // retrying in the case of signal or timeout
                close(fd);
                continue;
            }
            if (r != 0)
            {
                // acquire failed
                close(fd);
                return false;
            }
            // we own the exclusive lock on file descriptor, but we need to double-check
            // that the lock file wasn't deleted and/or re-created;
            // checking this by comparing inode numbers
            struct stat before, after;
            fstat(fd, &before);
            if (stat(m_fileName.c_str(), &after) != 0 || before.st_ino != after.st_ino)
            {
                // we have a race with 'unlink' call in Release()
                // our lock is held to the previous instance of the file;
                // this is not a problem, we just need to retry locking the new file
                close(fd);
                continue;
            }
            else
            {
                // lock acquired successfully
                m_fd = fd;
                return true;
            }
        }
    }

    // Releases the mutex
    void Release()
    {
        assert(m_fd != -1);
        // removing file
        unlink(m_fileName.c_str());
        // Note: file is intentionally removed *before* releasing the lock
        // to ensure that locked file isn't deleted by the non-owner of the lock
        m_lock.l_type = F_UNLCK;
        // Now removing the lock and closing the file descriptor
        // waiting processes will be notified
        int rc = fcntl(m_fd, F_SETLKW, &m_lock);
        if (rc == -1) {
            RuntimeError("Release: Failed to release mutex %S", m_fileName.c_str());
        }
        close(m_fd);
        m_fd = -1;
    }

    ~CrossProcessMutex()
    {
        if (m_fd != -1)
        {
            Release();
        }
    }
};

#endif
