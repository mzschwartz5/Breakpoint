#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Imath::Imath" for configuration "Release"
set_property(TARGET Imath::Imath APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Imath::Imath PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/Imath-3_1.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/Imath-3_1.dll"
  )

list(APPEND _cmake_import_check_targets Imath::Imath )
list(APPEND _cmake_import_check_files_for_Imath::Imath "${_IMPORT_PREFIX}/lib/Imath-3_1.lib" "${_IMPORT_PREFIX}/bin/Imath-3_1.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
