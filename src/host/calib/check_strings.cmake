# Host API 10 (DESIGN §6c.3): every string value of a calibration file (gpu, date, version,
# clocks, notes) must fit its imd_perf_calib_t field (bytes, NUL included). The top-level
# CMakeLists.txt includes this for the shipped calibration (a value that does not fit would
# make the embedded calibration fail to parse at run time, and every perf query with a NULL
# calibration return INVALID_VALUE); ctest calib_strings_* runs it as a script:
#   cmake -DIMD_CALIB_FILE=<file> -P src/host/calib/check_strings.cmake
# FATAL_ERROR names the file, the line and the field. bench/perf_fit.py checks what it writes;
# ctest host_c_api and host_perf parse the whole shipped file.
if(NOT DEFINED IMD_CALIB_FILE)
  message(FATAL_ERROR "check_strings.cmake: set IMD_CALIB_FILE")
endif()
file(READ "${IMD_CALIB_FILE}" _imd_rest)
# Walked line by line with string(FIND): the text contains ';', so it is never a list.
set(_imd_line_no 0)
while(NOT "${_imd_rest}" STREQUAL "")
  string(FIND "${_imd_rest}" "\n" _imd_eol)
  if(_imd_eol EQUAL -1)
    set(_imd_line "${_imd_rest}")
    set(_imd_rest "")
  else()
    string(SUBSTRING "${_imd_rest}" 0 ${_imd_eol} _imd_line)
    math(EXPR _imd_next "${_imd_eol} + 1")
    string(SUBSTRING "${_imd_rest}" ${_imd_next} -1 _imd_rest)
  endif()
  math(EXPR _imd_line_no "${_imd_line_no} + 1")
  foreach(_imd_field gpu:64 date:32 version:64 clocks:128 notes:128)
    string(REPLACE ":" ";" _imd_kc "${_imd_field}")
    list(GET _imd_kc 0 _imd_key)
    list(GET _imd_kc 1 _imd_cap)
    if("${_imd_line}" MATCHES "^[ \t]*${_imd_key}[ \t]*=(.*)$")
      string(STRIP "${CMAKE_MATCH_1}" _imd_val)
      string(LENGTH "${_imd_val}" _imd_len)
      if(NOT _imd_len LESS _imd_cap)
        math(EXPR _imd_max "${_imd_cap} - 1")
        message(FATAL_ERROR "${IMD_CALIB_FILE}:${_imd_line_no}: '${_imd_key}' is ${_imd_len} "
                            "bytes; the calibration field holds ${_imd_max} (DESIGN §6c.3)")
      endif()
    endif()
  endforeach()
endwhile()
