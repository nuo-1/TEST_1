find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDEDIR})
target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES})