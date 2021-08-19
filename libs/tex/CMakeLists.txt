file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")

add_library(tex ${SOURCES})
set_target_properties(tex PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION
        $<AND:$<BOOL:${CMAKE_CXX_LTO_SUPPORTED}>, $<BOOL:${ENABLE_IPO}>, $<NOT:$<CONFIG:Debug>>>
)
target_link_libraries(tex
    PUBLIC
        Eigen3::Eigen
        mve::core
    PRIVATE
        tex_settings
        rayint::acc
        mapmap
        $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>
)
target_include_directories(tex PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs>
)
