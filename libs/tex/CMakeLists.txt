file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")

add_library(tex STATIC ${SOURCES})
if (IPO_SUPPORTED)
    set_target_properties(tex PROPERTIES INTERPROCEDURAL_OPTIMIZATION True)
endif()
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
