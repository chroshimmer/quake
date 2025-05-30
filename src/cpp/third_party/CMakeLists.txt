include(ExternalProject)

function(initialize_submodule DIRECTORY)
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY}/.git)
        find_package(Git QUIET REQUIRED)
        message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY}/.git does not exist. Initializing ${DIRECTORY} submodule ...")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init ${DIRECTORY}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_EXIT_CODE)
        if(NOT GIT_EXIT_CODE EQUAL "0")
            message(FATAL_ERROR "${GIT_EXECUTABLE} submodule update --init dependencies/${DIRECTORY} failed with exit code ${GIT_EXIT_CODE}, please checkout submodules")
        endif()
    endif()
endfunction(initialize_submodule)

initialize_submodule(pybind11)
add_subdirectory(pybind11 EXCLUDE_FROM_ALL)

set(FAISS_ENABLE_PYTHON OFF)
set(FAISS_BUILD_TESTS OFF)
set(FAISS_BUILD_BENCHMARKS OFF)
set(FAISS_BUILD_EXAMPLES OFF)
initialize_submodule(faiss)
add_subdirectory(faiss)

initialize_submodule(concurrentqueue)
add_subdirectory(concurrentqueue EXCLUDE_FROM_ALL)