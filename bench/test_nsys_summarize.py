#!/usr/bin/env python3
"""Self-test of bench/nsys_summarize.py (CPU only: kernel-name lists and a synthetic nsys sqlite export).

    python3 bench/test_nsys_summarize.py

Checks the kernel classifier on names seen in real profiles (immerdink, cuTENSOR, cuBLAS, GEMMul8; the lists are
the regression test of the CATEGORY_RULES patterns), the timed-window logic (setup kernels outside the windows
are not counted, the first window is dropped, per-call averages), engaged_int8, the index.json / summary.json
files and the "nsys missing" path.
"""
import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsys_summarize as S  # noqa: E402

NAMES = {
    "int8_gemm": [
        # immerdink's INT8 GEMM mainloop
        "void imd::kernels::k4_kernel<imd::kernels::K4Gemm<cutlass::gemm::GemmShape<(int)128, (int)256, (int)64>, "
        "cutlass::gemm::GemmShape<(int)64, (int)64, (int)64>, (int)3, cutlass::arch::OpMultiplyAddSaturate>, "
        "(bool)0>(T1::Params)",
        # GEMMul8's INT8 GEMM (cuBLAS -> CUTLASS) on sm_89
        "void cutlass::Kernel2<cutlass_80_tensorop_i16832gemm_s8_256x128_64x3_tn_align16>(T1::Params)",
        # cuBLAS 13 FP64 emulation's INT8 GEMM (nvjet, type code biu), RTX 4090
        "nvjet_sm89_biu_mma_256x128x128_2_64x64x128_ldgstsAB_alignAB16_alignCD4_bz_dynbatch_smod_TNNN",
        "sm80_xmma_gemm_s8s8_s8s32_tn_n_tilesize128x128x64_stage3_warpsize2x2x1_tensor16x8x32_kernel",
        "ampere_i8816gemm_int8_128x128_ldg16_stages_64x3_tn",
        "turing_imma_s8_gemm_128x128",
        # GEMMul8's INT8 GEMM on H100 (cuBLAS wgmma kernel, R1 profile): was classified "other"
        "sm90_xmma_gemm_i8i32_i8i32_i32_tn_n_tilesize128x128x128_warpgroupsize1",
        "cutlass3x_sm90_tensorop_s64x128x32gemm_s8_s8_s32_s32_s32_128x128x128_0_tnn_align16_warpspecialized",
    ],
    "fp64_gemm": [
        # cuTENSOR native, cuBLAS DGEMM: seen on the RTX 4090
        "void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_128x128_16x3_tn_align1>(T1::Params)",
        "void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_64x32_16x4_nt_align1>(T1::Params)",
        "cutlass_sm80_tensorop_d884gett_f64_iden_f64_iden_f64_f64_f64_64x64x16_2x2x1_4x4x8_8x2_ttt_align1",
        "void cutlass::Kernel2<cublasLt_imma_emu::cutlass_80_tensorop_d884gemm_128x128_16x3_tn_align1>(T1::Params, const int *, int)",
        "ampere_dgemm_128x64_nn",
        "sm90_xmma_gemm_f64f64_f64f64_f64_tn_n_tilesize64x64x16_warpgroupsize1x1x1",
        "void cutlass::Kernel2<cutlass_80_tensorop_d16816gemm_64x64_16x4_nn_align1>(T1::Params)",
    ],
    "conversion_pack_crt": [
        "void imd::detail::<unnamed>::kl_scale_pack<(int)8, (int)2>(imd::detail::<unnamed>::LargeArgs)",
        "void imd::detail::<unnamed>::k5_crt_n<(int)14, (int)4>(imd::detail::<unnamed>::OutArgs)",
        "void imd::detail::<unnamed>::k0_vec<(int)8>(imd::detail::<unnamed>::K0Args<T1>)",
        "void gemmul8::scaling::fast::scaling_colwise<double, (gemmul8::Backend)0, (unsigned int)14>(unsigned int)",
        "void gemmul8::mod::mod_hi2group_kernel<(gemmul8::Backend)0, (unsigned int)14, (bool)0>(unsigned int)",
        "void gemmul8::undo_scaling::undo_scaling_kernel_special<double, (gemmul8::Backend)0>(unsigned int)",
        # cuBLAS FP64 emulation helpers: their template arguments say oz2_int8_dgemm, they are not GEMMs
        "void prolog_mod_dfma<oz2_int8_dgemm, (int)256, (int)3, (int)8, (int)0, (int)1>(long)",
        "void prolog_quantize_k_major_kernel<(bool)0, (bool)0, (int)256, double, short>(long, long)",
        "void inverse_scaling_mrc_kernel<oz2_int8_dgemm, (int)64, (scale_mode_t)4>(int)",
        "void reduce_nmoduli_kernel<oz2_int8_dgemm>(int, int, const int *)",
        "void sft_fast_k_major_l2<double, short, (int)64, (int)4, (bool)1, const int *>(int, int)",
        "void get_minmax_block_ker<(int)32, (int)4>(bool, bool, unsigned int)",
        "void cutlass_unfused_epilogue_ker<(int)16, (int)16>(cudaDataType_t, unsigned int)",
    ],
    "permute": [
        "void cutensor_internal_namespace::permute_coop<(int)80, double, double, double, int, cute::Layout<cute::tuple<>>>(...)",
        "void cutensor_internal_namespace::tensor_permute<(int)80, double, double, double, (cutensorOperator_t)1>(...)",
        "transpose_kernel<double>",
    ],
    "memcpy_memset": ["[CUDA memcpy HtoD]", "[CUDA memcpy DtoD]", "[CUDA memset]"],
    "other": [
        "k_unknown_thing", "void foo::bar_kernel<int>(int*)",
        # a small int32 SIMT batched GEMM of the cuBLAS emulation: "Transpose..." in its type names must not
        # make it a permutation
        "void cutlass::Kernel<cutlass::gemm::kernel::GemmBatched<cutlass::gemm::threadblock::MmaPipelined<"
        "cutlass::gemm::GemmShape<(int)32, (int)32, (int)8>, cutlass::transform::threadblock::RegularTileIterator<"
        "cutlass::MatrixShape<(int)32, (int)8>, int, cutlass::layout::ColumnMajor, (int)1, "
        "cutlass::transform::TransposePitchLinearThreadMapSimt<int>>>>>(T1::Params, const int *, int)",
    ],
}


def test_classifier():
    bad = []
    for cat, names in NAMES.items():
        for n in names:
            got = S.classify(n)
            if got != cat:
                bad.append((n[:80], cat, got))
    assert not bad, "misclassified: %r" % bad


def test_precedence():
    # a GEMM kernel with a conversion word in its name stays a GEMM; an INT8 GEMM is never fp64
    assert S.classify("cutlass_80_tensorop_s32_i16832gemm_s8_convert_scale") == "int8_gemm"
    # template arguments are not matched: a kernel is what its function name says
    assert S.classify("void foo_kernel<oz2_int8_dgemm, TransposeMap>(int)") == "other"
    assert S.classify("dgemm_scale_epilogue") == "fp64_gemm"
    assert S.classify("") == "other"


def make_db(path, with_windows=True, int8=True):
    """Synthetic export: setup kernel + 4 timed windows (one setup record before the first) of
    [int8 gemm 3 ms, pack 1 ms] each, plus an untimed native call after the windows."""
    if os.path.exists(path):
        os.remove(path)
    c = sqlite3.connect(path)
    c.executescript("""
        create table StringIds (id integer primary key, value text);
        create table CUPTI_ACTIVITY_KIND_KERNEL (start, end, deviceId, streamId, correlationId, globalPid,
                                                 demangledName, shortName);
        create table CUPTI_ACTIVITY_KIND_MEMCPY (start, end, copyKind, correlationId, globalPid);
        create table CUPTI_ACTIVITY_KIND_MEMSET (start, end, correlationId, globalPid);
        create table CUPTI_ACTIVITY_KIND_RUNTIME (start, end, nameId, correlationId, globalTid);
        create table TARGET_INFO_GPU (name text);
    """)
    names = {1: "k4_kernel_int8" if int8 else "ampere_dgemm_128x64_nn", 2: "kl_scale_pack", 3: "ref_kernel",
             4: "cudaEventRecord", 5: "cudaEventSynchronize", 6: "cudaLaunchKernel", 7: "cudaEventElapsedTime",
             8: "cutlass_80_tensorop_d884gemm"}
    c.executemany("insert into StringIds values (?, ?)", list(names.items()))
    c.execute("insert into TARGET_INFO_GPU values ('Fake GPU')")
    ms = 1_000_000
    t = 0
    k = lambda s, e, nid: c.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,0,7,0,1,?,?)", (s, e, nid, nid))
    rt = lambda s, e, nid: c.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,0,5)", (s, e, nid))
    k(t, t + 50 * ms, 3)            # setup / reference kernel: never counted
    t += 60 * ms
    c.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?,?,1,0,1)", (t, t + ms))   # setup H2D
    t += 5 * ms
    for w in range(4 if with_windows else 0):
        rt(t, t + 1000, 4)          # e0
        rt(t + 2000, t + 3000, 6)   # launch
        k(t + 4000, t + 4000 + 3 * ms, 1)
        k(t + 4000 + 3 * ms, t + 4000 + 4 * ms, 2)
        rt(t + 5000 + 4 * ms, t + 6000 + 4 * ms, 4)    # e1
        rt(t + 7000 + 4 * ms, t + 8000 + 5 * ms, 5)    # synchronize
        rt(t + 9000 + 5 * ms, t + 9500 + 5 * ms, 7)    # elapsed time
        t += 10 * ms
    k(t, t + 30 * ms, 8)            # untimed native call after the windows
    c.commit()
    c.close()


def test_windows_and_summary():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "r.sqlite")
        make_db(db)
        s = S.make_summary("tg", "case", "imd", 14, None, db)
        assert s["windows"] == 3, s["windows"]            # 4 windows, the first one dropped
        by = {k["name"]: k for k in s["kernels"]}
        assert set(by) == {"k4_kernel_int8", "kl_scale_pack"}, set(by)   # setup, memcpy, untimed call: out
        assert abs(by["k4_kernel_int8"]["time_ms"] - 3.0) < 1e-6 and by["k4_kernel_int8"]["count"] == 1
        assert abs(by["k4_kernel_int8"]["share_pct"] - 75.0) < 1e-6
        assert by["k4_kernel_int8"]["category"] == "int8_gemm" and by["kl_scale_pack"]["category"] == "conversion_pack_crt"
        assert s["engaged_int8"] is True and abs(s["int8_share_pct"] - 75.0) < 1e-6
        assert 4.0 < s["wall_ms"] < 5.5, s["wall_ms"]
        assert s["gpu"] == "Fake GPU"
        # native FP64 only: cutensor-emu is NOT engaged and the notes say so
        make_db(db, int8=False)
        s2 = S.make_summary("tg", "case", "cutensor-emu", None, None, db)
        assert s2["engaged_int8"] is True or s2["engaged_int8"] is False
        by2 = {k["name"]: k["category"] for k in s2["kernels"]}
        assert by2["ampere_dgemm_128x64_nn"] == "fp64_gemm"
        assert s2["engaged_int8"] is False and "NOT engaged" in s2["notes"], s2["notes"]


def test_no_window_fallback_and_empty():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "r.sqlite")
        make_db(db, with_windows=False)
        s = S.make_summary("tg", "case", "imd", 8, None, db)
        assert "whole process" in s["notes"] and s["windows"] == 1
        # no GPU activity at all: engaged_int8 is null
        c = sqlite3.connect(db)
        c.execute("delete from CUPTI_ACTIVITY_KIND_KERNEL")
        c.execute("delete from CUPTI_ACTIVITY_KIND_MEMCPY")
        c.commit()
        c.close()
        s = S.make_summary("tg", "case", "imd", 8, None, db)
        assert s["engaged_int8"] is None and s["kernels"] == []


def run_cli(*args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = S.main(list(args))
    return rc, buf.getvalue()


def test_cli_files():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "r.sqlite")
        make_db(db)
        out = os.path.join(d, "profile")
        rpt = os.path.join(d, "records.jsonl")
        with open(rpt, "w") as f:
            f.write(json.dumps({"method": "emu", "status": "error: needs CUDA 13 cuBLAS"}) + "\n")
        run_cli("summarize", "--db", db, "--tag", "a-imd-N14", "--case", "a", "--method", "imd", "--N", "14",
                "--outdir", out, "--nsys-rep", "a-imd-N14/a.nsys-rep", "--gpu", "G")
        run_cli("summarize", "--db", db, "--tag", "a-emu-dyn", "--case", "a", "--method", "emu", "--N", "-",
                "--variant", "dyn", "--outdir", out, "--runlog", rpt)
        run_cli("summarize", "--db", db, "--tag", "a-imd-N14", "--case", "a", "--method", "imd", "--N", "14",
                "--outdir", out)   # re-run replaces the entry
        idx = json.load(open(os.path.join(out, "index.json")))
        assert [e["tag"] for e in idx] == ["a-emu-dyn", "a-imd-N14"], idx
        e = [x for x in idx if x["tag"] == "a-emu-dyn"][0]
        assert e["N"] is None and e["variant"] == "dyn" and e["summary"] == "a-emu-dyn/summary.json"
        sm = json.load(open(os.path.join(out, e["summary"])))
        for key in ("tag", "case", "method", "N", "gpu", "wall_ms", "kernels", "engaged_int8", "notes"):
            assert key in sm, key
        assert "needs CUDA 13 cuBLAS" in sm["notes"] and sm["bench_status"].startswith("error")
        for k in sm["kernels"]:
            assert set(k) == {"name", "category", "time_ms", "share_pct", "count"}
            assert k["category"] in S.CATEGORIES
        assert "a-imd-N14" in run_cli("table", out)[1]


def test_missing():
    with tempfile.TemporaryDirectory() as d:
        ent = os.path.join(d, "e.tsv")
        with open(ent, "w") as f:
            f.write("a-imd-N14\ta\timd\t14\t-\na-gemmul8-fast-N14\ta\tgemmul8\t14\tfast\na-cutensor\ta\tcutensor\t-\t-\n")
        out = os.path.join(d, "profile")
        run_cli("missing", "--outdir", out, "--nsys-note", "nsys not found", "--entries", ent)
        idx = json.load(open(os.path.join(out, "index.json")))
        assert len(idx) == 3 and all(e["engaged_int8"] is None and e["nsys_rep"] is None for e in idx)
        by = {e["tag"]: e for e in idx}
        assert by["a-cutensor"]["N"] is None and by["a-cutensor"]["variant"] is None
        assert by["a-gemmul8-fast-N14"]["variant"] == "fast" and by["a-imd-N14"]["N"] == 14
        sm = json.load(open(os.path.join(out, "a-imd-N14", "summary.json")))
        assert sm["engaged_int8"] is None and sm["kernels"] == [] and "nsys not found" in sm["notes"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except AssertionError as e:
                fails += 1
                print("FAIL ", name, e)
    print("%s" % ("FAILED" if fails else "all passed"))
    sys.exit(1 if fails else 0)
