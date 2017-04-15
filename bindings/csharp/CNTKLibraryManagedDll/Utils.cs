//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Util.cs -- Implements CNTK utility functions
//

namespace CNTK
{
    public class Utils
    {
        /// <summary>
        /// Sets the process-wide maximum number of CPU threads to be used by any individual compute operation.
        /// Note that this is a per compute operation limit and if the user performs multiple compute operations concurrently
        /// by launching multiple threads and performing a compute operation inside, it will result in each of those concurrently
        /// executing operations to use the specified number of CPU threads limit.
        /// </summary>
        /// <param name="numCPUThreads">The maximum number of CPU threads</param>
        public static void SetMaxNumCPUThreads(int numCPUThreads)
        {
            if (numCPUThreads <= 0)
            {
                throw new System.ArgumentException("The maximum number of CPU threads, numCPUThreads, must be at least 1.");
            }
            CNTKLib.SetMaxNumCPUThreads((uint)numCPUThreads);
        }

        /// <summary>
        /// Returns the process-wide maximum number of CPU threads to be used by any individual compute operation.
        /// </summary>
        /// <returns>The current maximum number of CPU threads to be used by any individual compute operation.</returns>
        public static int GetMaxNumCPUThreads()
        {
            return (int)CNTKLib.GetMaxNumCPUThreads();
        }

        /// <summary>
        /// Specifies global logging verbosity level.
        /// </summary>
        /// <param name="value">The verbosity level: Error, Warning or Info.</param>
        public static void SetTraceLevel(TraceLevel value)
        {
            CNTKLib.SetTraceLevel(value);
        }

        /// <summary>
        /// Returns current logging verbosity level.
        /// </summary>
        /// <returns>The current verbosity level</returns>
        public static TraceLevel GetTraceLevel()
        {
            return CNTKLib.GetTraceLevel();
        }

        /// <summary>
        /// Returns the name of the given data type.
        /// </summary>
        /// <param name="dataType">The data type whose name is to be returned.</param>
        /// <returns></returns>
        public static string DataTypeName(DataType dataType)
        {
            return CNTKLib.DataTypeName(dataType);
        }

        /// <summary>
        /// Returns the size in byte of the given data type.
        /// </summary>
        /// <param name="dataType">The data type whose size is to be returned.</param>
        /// <returns></returns>
        public static int DataTypeSize(DataType dataType)
        {
            return (int)CNTKLib.DataTypeSize(dataType);
        }

        /// <summary>
        /// Returns true if the given storage format is a sparse format. Otherwise returns false.
        /// </summary>
        /// <param name="dataType">The storage format to be checked.</param>
        /// <returns></returns>
        public static bool IsSparseStorageFormat(StorageFormat storageFormat)
        {
            return CNTKLib.IsSparseStorageFormat(storageFormat);
        }

        /// <summary>
        /// Returns the name of the given device kind.
        /// </summary>
        /// <param name="deviceKind">The device kind whose name is to be returned.</param>
        /// <returns></returns>
        public static string DeviceKindName(DeviceKind deviceKind)
        {
            return CNTKLib.DeviceKindName(deviceKind);
        }

        /// <summary>
        /// Returns the name of the given variable kind.
        /// </summary>
        /// <param name="variableKind">The variable kind whose name is to be returned.</param>
        /// <returns></returns>
        public static string VariableKindName(VariableKind variableKind)
        {
            return CNTKLib.VariableKindName(variableKind);
        }
    }
}
