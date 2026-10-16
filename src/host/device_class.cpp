// Device classes, table-set selection and plan status (host API 14, docs/NOTES-18.md).
// CUDA-free: every function is a pure function of its arguments.
#include <cstring>

#include "immerdink/immerdink_host.h"

namespace {

// The built-in registry: the cards each table set was fitted on. A row is added together with
// the data that fitted it (docs/NOTES-18.md D2).
constexpr imd_table_row_t kRows[] = {
    {IMD_TABLE_SM89, 8, 9, 128, "NVIDIA GeForce RTX 4090"},
};
constexpr int32_t kRowCount = static_cast<int32_t>(sizeof kRows / sizeof kRows[0]);

bool has_nul(const char* s, size_t cap) { return std::memchr(s, '\0', cap) != nullptr; }

bool starts_with(const char* name, const char* prefix) {
  return std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

int32_t dev_version(int32_t major, int32_t minor) { return 10 * major + minor; }

}  // namespace

extern "C" {

int32_t imd_device_row_count(void) { return kRowCount; }

imd_status_t imd_device_row(int32_t index, imd_table_row_t* out) {
  if (out == nullptr || index < 0 || index >= kRowCount) return IMD_STATUS_INVALID_VALUE;
  *out = kRows[index];
  return IMD_STATUS_SUCCESS;
}

imd_status_t imd_device_select_rows(const imd_device_class_t* d, const imd_table_row_t* rows,
                                    int32_t num_rows, imd_device_selection_t* out) {
  if (d == nullptr || out == nullptr || out->struct_size < sizeof(imd_device_selection_t) ||
      d->struct_size < sizeof(imd_device_class_t) || num_rows < 0 ||
      (num_rows > 0 && rows == nullptr) || d->cc_major < 0 || d->cc_minor < 0 || d->sm_count < 0 ||
      !has_nul(d->name, sizeof d->name))
    return IMD_STATUS_INVALID_VALUE;
  for (int32_t i = 0; i < num_rows; ++i)
    if (rows[i].table_id < 0 || rows[i].table_id >= IMD_TABLE_COUNT || rows[i].cc_major < 0 ||
        rows[i].cc_minor < 0 || rows[i].sm_count < 0 || !has_nul(rows[i].name_prefix, sizeof rows[i].name_prefix))
      return IMD_STATUS_INVALID_VALUE;
  imd_device_selection_t r{};
  r.struct_size = out->struct_size;
  r.table_id = -1, r.row_index = -1;
  const auto take = [&](int32_t match, int32_t row) {
    r.match = match, r.row_index = row, r.table_id = rows[row].table_id;
    r.approximate = match == IMD_MATCH_EXACT ? 0 : 1;
  };
  if (dev_version(d->cc_major, d->cc_minor) < 80 || num_rows == 0) {
    r.match = IMD_MATCH_UNSUPPORTED, r.approximate = 1;
  } else {
    int32_t same_cc = -1, gen = -1, exact = -1, def = -1;
    for (int32_t i = 0; i < num_rows; ++i) {
      const imd_table_row_t& w = rows[i];
      if (def < 0 && w.cc_major == 8 && w.cc_minor == 9) def = i;
      if (w.cc_major == d->cc_major && w.cc_minor == d->cc_minor) {
        if (same_cc < 0) same_cc = i;
        if (exact < 0 && (w.sm_count == 0 || w.sm_count == d->sm_count) &&
            starts_with(d->name, w.name_prefix))
          exact = i;
      } else if (w.cc_major == d->cc_major) {
        // nearest minor, ties to the lower minor, then the earlier row
        const auto dist = [&](const imd_table_row_t& x) {
          const int32_t dm = x.cc_minor - d->cc_minor;
          return dm < 0 ? -dm : dm;
        };
        if (gen < 0 || dist(w) < dist(rows[gen]) ||
            (dist(w) == dist(rows[gen]) && w.cc_minor < rows[gen].cc_minor))
          gen = i;
      }
    }
    if (exact >= 0) take(IMD_MATCH_EXACT, exact);
    else if (same_cc >= 0) take(IMD_MATCH_ARCH, same_cc);
    else if (gen >= 0) take(IMD_MATCH_GENERATION, gen);
    else take(IMD_MATCH_DEFAULT, def >= 0 ? def : 0);
  }
  std::memcpy(out, &r, sizeof r);
  return IMD_STATUS_SUCCESS;
}

imd_status_t imd_device_select(const imd_device_class_t* d, imd_device_selection_t* out) {
  return imd_device_select_rows(d, kRows, kRowCount, out);
}

int32_t imd_kernel_code_kind(int32_t err, int32_t binary, int32_t ptx, int32_t major, int32_t minor) {
  if (err != 0) return IMD_CODE_UNUSABLE;
  const int32_t dev = dev_version(major, minor);
  if (binary == ptx) {
    if (binary == dev) return IMD_CODE_NATIVE;
    // prebuilt code of an older minor version of the same major runs as it is; anything else
    // (a newer or another major's SASS) cannot be what the device is running
    if (binary < dev && binary / 10 == major) return IMD_CODE_COMPATIBLE;
    return IMD_CODE_UNUSABLE;
  }
  // The driver compiled PTX (version ptx) for this device (binary): only plausible if the
  // PTX is not newer than the device.
  return binary == dev && ptx < dev ? IMD_CODE_RECOMPILED : IMD_CODE_UNUSABLE;
}

int32_t imd_plan_status_of(int32_t match, int32_t code) {
  if (match == IMD_MATCH_UNSUPPORTED || code == IMD_CODE_UNUSABLE) return IMD_PLAN_UNSUPPORTED;
  if (code == IMD_CODE_RECOMPILED) return IMD_PLAN_RECOMPILED;
  if (match != IMD_MATCH_EXACT || code == IMD_CODE_COMPATIBLE) return IMD_PLAN_APPROXIMATE;
  return IMD_PLAN_NATIVE;
}

const char* imd_match_name(int32_t v) {
  switch (v) {
    case IMD_MATCH_EXACT: return "exact";
    case IMD_MATCH_ARCH: return "arch";
    case IMD_MATCH_GENERATION: return "generation";
    case IMD_MATCH_DEFAULT: return "default";
    case IMD_MATCH_UNSUPPORTED: return "unsupported";
    default: return "?";
  }
}

const char* imd_code_kind_name(int32_t v) {
  switch (v) {
    case IMD_CODE_NATIVE: return "native";
    case IMD_CODE_COMPATIBLE: return "compatible";
    case IMD_CODE_RECOMPILED: return "recompiled";
    case IMD_CODE_UNUSABLE: return "unusable";
    default: return "?";
  }
}

const char* imd_plan_status_name(int32_t v) {
  switch (v) {
    case IMD_PLAN_NATIVE: return "native";
    case IMD_PLAN_APPROXIMATE: return "approximate";
    case IMD_PLAN_RECOMPILED: return "recompiled";
    case IMD_PLAN_UNSUPPORTED: return "unsupported";
    default: return "?";
  }
}

}  // extern "C"
