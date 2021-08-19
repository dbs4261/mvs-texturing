file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")

add_executable(texrecon ${SOURCES})
set_target_properties(texrecon PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION
        $<AND:$<BOOL:${CMAKE_CXX_LTO_SUPPORTED}>, $<BOOL:${ENABLE_IPO}>, $<NOT:$<CONFIG:Debug>>>
)
target_link_libraries(texrecon tex mve::core $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX> tex_settings)
install(TARGETS texrecon
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
