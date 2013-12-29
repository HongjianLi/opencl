OpenCL Samples
==============

deviceQuery
-----------

This sample enumerates the properties of the OpenCL devices present in the system.

bandwidthTest
-------------

This sample measures host to device and device to host copy bandwidth for pageable and page-locked memory of transfer sizes 3KB, 15KB, 15MB and 100MB via clEnqueueReadBuffer/clEnqueueWriteBuffer, clEnqueueMapBuffer/clEnqueueUnmapMemObject and clEnqueueCopyBuffer, and outputs them in CSV format.

jit
---

This sample jit-in-time compiles a .cl and outputs build log.

vectorAdd
---------

This sample uses async API and constant and shared memory to add two vectors of float.

mwc64x
------

This sample features a random number generator based on multiply-with-carry.

multiDevice
-----------

This sample distributes multiple tasks to multiple devices dynamically, and uses asynchronous callbacks for postprocessing.
