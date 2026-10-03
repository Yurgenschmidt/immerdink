// Visibility of libimmerdink_host's internal C++ entry points (DESIGN §6a).
//
// libimmerdink_host is built with hidden visibility. Its public interface is the C ABI of
// include/immerdink/immerdink_host.h. The C++ functions marked IMD_HOST_INTERNAL are
// exported as well, for libimmerdink (the CUDA library, which links this library and uses
// them for its plans) and for the unit tests only: they are not part of the public API, are
// not declared in include/, and carry no ABI stability guarantee (both libraries are built
// together, by the same compiler).
#pragma once

#if defined(_WIN32)
#  define IMD_HOST_INTERNAL __declspec(dllexport)
#else
#  define IMD_HOST_INTERNAL __attribute__((visibility("default")))
#endif
