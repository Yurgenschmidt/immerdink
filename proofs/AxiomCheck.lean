import Proofs

/-! Run by tools/check_proofs.sh: every headline theorem must depend only on
propext / Classical.choice / Quot.sound. -/

open Immerdink

-- L2 (CRT.lean)
#print axioms CRT.x_lt_P
#print axioms CRT.xp_lt_two_P
#print axioms CRT.xp_lt_word
#print axioms CRT.x_N_modEq
#print axioms CRT.exists_q
#print axioms CRT.z_eq
#print axioms CRT.z_eq_of_even
#print axioms CRT.z_eq_floor

-- L4 (Chunk.lean)
#print axioms Chunk.satFold_eq
#print axioms Chunk.AccTree.eval_eq_sum
#print axioms Chunk.partial_sum_bound
#print axioms Chunk.satFold_chunk_eq_sum
#print axioms Chunk.tree_chunk_eq_sum
#print axioms Chunk.epi_modEq_total
#print axioms Chunk.epi_mem_range
#print axioms Chunk.carry_correct
#print axioms Chunk.epi_input_range
#print axioms Chunk.design_constants
#print axioms Chunk.epi_input_int32

-- L1 (Scaling.lean)
#print axioms Scaling.l2_add_le
#print axioms Scaling.l1_le_sqrt_mul_l2
#print axioms Scaling.two_pow_le_S
#print axioms Scaling.exists_max_rule
#print axioms Scaling.not_rule_succ
#print axioms Scaling.exists_u_rn
#print axioms Scaling.exists_u_trunc
#print axioms Scaling.no_rule_of_R_zero
#print axioms Scaling.budget_rn
#print axioms Scaling.budget_trunc
#print axioms Scaling.budget_rn_int
#print axioms Scaling.budget_trunc_int
#print axioms Scaling.crt_precondition
#print axioms Scaling.sigma_bound_rn
#print axioms Scaling.sigma_bound_trunc
#print axioms Scaling.row_rn
#print axioms Scaling.row_trunc
#print axioms Scaling.abs_rne_sub
#print axioms Scaling.abs_truncZ_le
#print axioms Scaling.abs_truncZ_sub_lt

-- L3 (Error.lean)
#print axioms Error.l1_form
#print axioms Error.normwise_core
#print axioms Error.normwise_rn
#print axioms Error.normwise_trunc
#print axioms Error.end_to_end_rn
#print axioms Error.end_to_end_trunc
#print axioms Error.zero_row
#print axioms Error.q_zero_rn
-- L3c (Error.lean; NOTES-C1 IR5)
#print axioms Error.l3c_unit
#print axioms Error.round_step
#print axioms Error.l3c_normwise
#print axioms Error.l3c_elementwise
#print axioms Error.l3c_absolute
#print axioms Error.unitE_mono

-- L7a recursive summation, L3c (3) native chain (Summation.lean)
#print axioms Summation.pow_mul_le_one
#print axioms Summation.pow_sub_one_le_gam
#print axioms Summation.gam_nonneg
#print axioms Summation.recsum_pow
#print axioms Summation.recsum
#print axioms Summation.fma_chain

-- L7 segmented accumulation (Segments.lean)
#print axioms Segments.band_bound
#print axioms Segments.kseg_compose_sub
#print axioms Segments.kseg_compose
#print axioms Segments.abs_dot_le
#print axioms Segments.kseg_rows
#print axioms Segments.bands_norm_le
#print axioms Segments.kseg_normwise
#print axioms Segments.gam_mul_le
#print axioms Segments.kseg_request_core
#print axioms Segments.kseg_wbar_le
#print axioms Segments.kseg_rho_le
#print axioms Segments.kseg_rho_le_sub
#print axioms Segments.kseg_wbar
#print axioms Segments.kseg_elementwise
#print axioms Segments.kseg_absolute

-- Capstone (Exact.lean): L1c + L4 + L2c
#print axioms Exact.emulated_product_exact

-- DESIGN §3.2 the shift rule as computed (ShiftRule.lean)
#print axioms Scaling.kappa_isqrt
#print axioms Scaling.shift_estimate
#print axioms Scaling.u_eq_estimate
#print axioms Scaling.S_le
#print axioms Scaling.S_lt_two_pow_128

-- DESIGN §3.4-§3.6 residues and exact reductions (Residue.lean)
#print axioms Residue.smod_modEq
#print axioms Residue.smod_congr
#print axioms Residue.smod_range
#print axioms Residue.smod_int8
#print axioms Residue.smod_mul_abs_le
#print axioms Residue.int8_mul_ge
#print axioms Residue.smod_mul_ge
#print axioms Residue.smod_table
#print axioms Residue.smod_neg_odd
#print axioms Residue.smod_neg_byte
#print axioms Residue.byte_weights_modEq
#print axioms Residue.byte_weights_le
#print axioms Residue.mulhi_div_eq
#print axioms Residue.mod_p_bytes32
#print axioms Residue.barrett_epilogue
#print axioms Residue.crt_y_reduction
#print axioms Residue.emulated_product_exact_smod

-- DESIGN §3.6 the CRT word: peak, compare, wrap, centring (CRTWord.lean)
#print axioms CRT.R_dvd_x
#print axioms CRT.x_le_P_sub_R
#print axioms CRT.xp_le_peak
#print axioms CRT.xp_le_crtPeak
#print axioms CRT.xp_ne_P
#print axioms CRT.le_xp_iff_lt
#print axioms CRT.xp_mod_word
#print axioms CRT.center_word
#print axioms CRT.z_word
#print axioms CRT.peak_attained
#print axioms CRT.crtPeak_attained

-- DESIGN §2 the design instance (Moduli.lean)
#print axioms Moduli.pm_pos
#print axioms Moduli.pm_le_256
#print axioms Moduli.pm_coprime
#print axioms Moduli.P_even
#print axioms Moduli.exists_q_design
#print axioms Moduli.T_ge_180
#print axioms Moduli.L32_exact
#print axioms Moduli.peak_needs_every_limb
#print axioms Moduli.chunk_constants
#print axioms Moduli.chunk_sum_range
#print axioms Moduli.epilogue_range_v113
#print axioms Moduli.emulated_product_exact_design

-- L5 K-weights (KWeights.lean; NOTES-C1 §4.4)
#print axioms KWeights.kweight_cancel
#print axioms KWeights.l2_mul_const
#print axioms KWeights.norm_le_of_rule
#print axioms KWeights.kweight_ratio
#print axioms KWeights.kweight_ratio_rn
#print axioms KWeights.kweight_ratio_trunc
#print axioms KWeights.kweight_guard
#print axioms KWeights.sF_sq_le
#print axioms KWeights.kweight_guard_lt
#print axioms KWeights.kweight_transfer_elementwise
#print axioms KWeights.kweight_transfer_absolute
#print axioms KWeights.kweight_transfer_normwise

-- L6 a-posteriori certificate (Apost.lean; NOTES-C1 §4.4)
#print axioms Apost.apost_bound
#print axioms Apost.apost_scaled
#print axioms Apost.apost_rho
#print axioms Apost.apost_elementwise
#print axioms Apost.apost_pow2
