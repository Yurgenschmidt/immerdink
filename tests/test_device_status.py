#!/usr/bin/env python3
"""Device status on the GPU this runs on (issue 18, docs/NOTES-18.md): imd_device_info, imd_plan_info.

    python3 tests/test_device_status.py [--lib build/libimmerdink.so] [--expect STATUS[,STATUS]] [--expect-error CODE[,CODE]]

Checks that hold on any card: the device class matches the CUDA runtime's; the selection is what the
CUDA-free rule gives for that class; the code kind and status agree with the CUDA-free classification
rules; every kernel was checked; the answer is cached (a second call is fast); a plan carries the same
device, table set and status plus its K4 configuration, and imd_lowering for the plan's SM count and
table set reports that configuration; invalid arguments are rejected. With --expect (or on a card
the built-in registry knows, an RTX 4090 with a native build: native) it also pins the status.
Environment CUDA_MODULE_LOADING=EAGER runs the same checks (ctest device_status_eager).
"""
import argparse
import ctypes as C
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402


class Fail(Exception):
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--expect", help="comma list of acceptable statuses: native, approximate, recompiled, unsupported")
    ap.add_argument("--expect-error", help="comma list of acceptable CUDA errors when the status is unsupported "
                                           "(209 no kernel image, 222 PTX newer than the driver)")
    args = ap.parse_args()
    os.environ["IMMERDINK_LIB_DIR"] = os.path.dirname(os.path.abspath(args.lib))
    import immerdink
    from immerdink import host

    try:
        t0 = time.time()
        info = immerdink.device_info()
        t_first = time.time() - t0
        t0 = time.time()
        again = immerdink.device_info()
        t_again = time.time() - t0
        if again != info:
            raise Fail("device_info changed between two calls")
        if t_again > 0.01 and t_again > t_first / 5:
            raise Fail(f"the second call took {t_again:.4f} s (first {t_first:.4f} s): not cached")
        # a plan of an empty GEMM touches no CUDA: it is created even on a device the library cannot use, and its info is zero
        np_ = immerdink.Plan("ik,kj->ij", immerdink.Layout((0, 8)), immerdink.Layout((8, 8)), immerdink.Layout((0, 8)), N=8)
        if np_.info.kernels_checked != 0 or np_.info.sm_count != 0:
            raise Fail(f"an empty plan carries device info: {np_.info}")
        # the device class against the runtime
        lib = immerdink.load_library()
        dev = lib.current_device()
        if info.sm_count != lib.sm_count(dev):
            raise Fail(f"sm_count {info.sm_count} != the runtime's {lib.sm_count(dev)}")
        if info.k4_config != -1:
            raise Fail("device_info.k4_config must be -1")
        # the selection is the CUDA-free rule's
        sel = host.device_select(info.cc, info.sm_count, info.name)
        if (sel.match, sel.table_id, sel.approximate) != (info.match, info.table_id, info.approximate) \
                and info.match != "unsupported":
            raise Fail(f"selection {info.match}/{info.table_id} differs from the rule's {sel}")
        # status = summary of match and code kind
        if info.status == "native" and (info.match != "exact" or info.code_kind != "native"):
            raise Fail(f"status native with {info.match} tables and {info.code_kind} code")
        if info.status == "unsupported" and info.code_kind != "unusable" and info.match != "unsupported":
            raise Fail("unsupported without a reason")
        if info.code_kind == "unusable" and (info.failing_kernel < 0 or info.cuda_error == 0):
            raise Fail("unusable code without the failing kernel and CUDA error")
        if info.status != "unsupported" and info.kernels_checked < 100:
            raise Fail(f"only {info.kernels_checked} kernels checked")
        if args.expect and info.status not in args.expect.split(","):
            raise Fail(f"status {info.status}, expected {args.expect}")
        if args.expect_error and info.status == "unsupported" and \
                str(info.cuda_error) not in args.expect_error.split(","):
            raise Fail(f"CUDA error {info.cuda_error}, expected {args.expect_error}")
        if not args.expect and info.name.startswith("NVIDIA GeForce RTX 4090") and info.sm_count == 128 \
                and info.status != "native":
            raise Fail(f"an RTX 4090 gets status {info.status}")
        if info.status == "unsupported":
            # plans must refuse, with NOT_SUPPORTED, and say why
            try:
                immerdink.Plan("ik,kj->ij", *[immerdink.Layout((8, 8))] * 3, N=8)
            except immerdink.NotSupportedError:
                pass
            else:
                raise Fail("a plan was created on an unsupported device")
            print(f"PASS  the device is unsupported by this build: CUDA error {info.cuda_error} "
                  f"({info.match} tables), first failing kernel {info.failing_kernel}")
            return 0
        # a plan
        L_ = immerdink.Layout
        for spec, shapes, N in (("ik,kj->ij", ((64, 96), (96, 80), (64, 80)), 8),
                                ("ik,kj->ij", ((300, 400), (400, 500), (300, 500)), 14),
                                ("i,i->", ((16,), (16,), ()), 8)):
            p = immerdink.Plan(spec, *[L_(sh) for sh in shapes], N=N)
            pi = p.info
            same = (pi.name, pi.cc, pi.sm_count, pi.match, pi.table_id, pi.approximate, pi.code_kind, pi.status,
                    pi.kernels_checked)
            want = (info.name, info.cc, info.sm_count, info.match, info.table_id, info.approximate, info.code_kind,
                    info.status, info.kernels_checked)
            if same != want:
                raise Fail(f"plan info {same} differs from the device's {want}")
            lw = p.lowering
            if lw.table_id != info.table_id or lw.sm_count != info.sm_count:
                raise Fail("the plan's lowering was not made for the plan's device and table set")
            if lw.dispatch == "EMULATE" and pi.k4_config != lw.k4_config:
                raise Fail(f"plan K4 configuration {pi.k4_config} != the rule's {lw.k4_config}")
            if lw.dispatch == "EMULATE" and not 0 <= pi.k4_config < 4:
                raise Fail(f"an emulated plan reports K4 configuration {pi.k4_config}")
            p.close()
        # invalid arguments
        L = lib.lib
        bad = immerdink._capi.DeviceInfo()
        bad.struct_size = 8
        if L.imd_device_info(immerdink.default_handle()._h, C.byref(bad)) != 1 or \
                L.imd_device_info(None, C.byref(bad)) != 1 or L.imd_plan_info(None, C.byref(bad)) != 1:
            raise Fail("device_info / plan_info must reject a short struct and NULL")
    except Fail as e:
        print(f"FAIL  {e}")
        return 1
    print(f"PASS  {info.name} sm_{info.cc[0]}{info.cc[1]}, {info.sm_count} SMs: tables {info.match}, code "
          f"{info.code_kind}, status {info.status}; {info.kernels_checked} kernels checked in {t_first * 1000:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
